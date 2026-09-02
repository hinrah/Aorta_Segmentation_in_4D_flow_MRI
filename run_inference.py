"""
Cascaded PCMRA-based segmentation pipeline.

Behaves like nnUNetv2_predict from the outside (input folder -> output folder),
but internally:
  1. Creates the output folder and a temporary folder inside it.
  2. Builds a PCMRA image (magnitude[_0000] * magnitude[_0000] * velocity_magnitude[_0001] * velocity_magnitude[_0001],
     averaged over time) for every case and stores it in the temp folder.
  3. Segments the PCMRA images with Dataset100 (localization).
  4. Crops the original input images to the bounding box of the PCMRA
     segmentation.
  5. Segments the cropped images with Dataset101 (fine segmentation).
  6. Un-crops the result back into the original image space and saves it.
"""

import argparse
import os
import shutil
import warnings
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder

SAGITTAL_MAX_TILT_DEG = 45.0

def _warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def check_sagittal_orientation(affine, case, name):

    slice_dim = 2

    normal = np.asarray(affine[:3, slice_dim], dtype=np.float64)
    norm = np.linalg.norm(normal)
    normal /= norm

    cos_tilt = abs(float(normal[0]))
    tilt_deg = float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))

    if tilt_deg > SAGITTAL_MAX_TILT_DEG:
        axcodes = "".join(nib.aff2axcodes(affine))
        _warn(f"[{case}/{name}] Dataset does not appear to be acquired sagittally: "
              f"slice-normal tilt vs. left-right axis is {tilt_deg:.1f}° "
              f"(allowed max. {SAGITTAL_MAX_TILT_DEG:.0f}°, orientation {axcodes}). "
              f"Prediction quality may be degraded.")
        return False
    return True


def check_spacing(header, case, name):
    zooms = np.asarray(header.get_zooms()[:3], dtype=np.float64)
    if np.allclose(zooms, 1.0, atol=1e-6):
        _warn(f"[{case}/{name}] Voxel spacing is exactly 1.0 x 1.0 x 1.0 mm, is this correct or nifti default spacing?"
              f"The correct spacing must be stored in the NIfTI header - "
              f"otherwise resampling will be corrupted and the results invalid.")
        return False
    return True


def check_input_geometry(affine, header, case, name):
    ok_orient = check_sagittal_orientation(affine, case, name)
    ok_spacing = check_spacing(header, case, name)
    return ok_orient and ok_spacing

def find_cases(input_folder):
    cases = []
    for f in os.listdir(input_folder):
        if f.endswith("_0000.nii.gz"):
            cases.append(f[: -len("_0000.nii.gz")])
    return sorted(cases)


def load_nifti(path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def create_pcmra(magnitude, velocity):
    magnitude = 255*magnitude/np.max(magnitude)
    pcmra_4d = magnitude.squeeze() * magnitude.squeeze() * velocity.squeeze() * velocity.squeeze()
    if pcmra_4d.ndim == 4:
        pcmra = pcmra_4d.mean(axis=-1)
    else:
        pcmra = pcmra_4d

    return pcmra.astype(np.float32)


def bounding_box(mask, margin=0):
    binary = mask > 0
    if not binary.any():
        raise RuntimeError("PCMRA segmentation is empty; cannot compute bbox.")

    # coordinates of all nonzero voxels
    coords = np.array(np.nonzero(binary))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1

    slices = []
    for i in range(mask.ndim):
        lo = max(0, mins[i] - margin)
        hi = min(mask.shape[i], maxs[i] + margin)
        slices.append(slice(lo, hi))
    return tuple(slices)


def build_predictor(dataset, configuration, folds, checkpoint,
                    nnunet_results, device, plans, trainer):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    model_folder = get_output_folder(dataset, trainer, plans, configuration)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=checkpoint,
    )
    return predictor


def main():
    parser = argparse.ArgumentParser(
        description="Cascaded PCMRA segmentation (nnUNetv2_predict-like interface)."
    )
    parser.add_argument("-i", type=str, required=True,
                        help="Input folder (contains *_0000.nii.gz and *_0001.nii.gz).")
    parser.add_argument("-o", type=str, required=True,
                        help="Output folder.")
    parser.add_argument("--c_loc", type=str, default="3d_fullres",
                        help="nnUNet configuration (default: 3d_fullres).")
    parser.add_argument("--c_seg", type=str, default="4d_hybrid_kernel",
                        help="nnUNet configuration (default: 4d_hybrid_kernel).")
    parser.add_argument("-f", nargs="+", type=str, default=["0", "1", "2", "3", "4"],
                        help="Folds to use (default: 0, 1, 2, 3, 4).")
    parser.add_argument("--tr_loc", type=str, default="nnUNetTrainer",
                        help="Trainer name.")
    parser.add_argument("--tr_seg", type=str, default="nnUNetTrainer_500epochs_lre3",
                            help="Trainer name.")
    parser.add_argument("-p", type=str, default="nnUNetPlans",
                        help="Plans identifier.")
    parser.add_argument("-chk", type=str, default="checkpoint_final.pth",
                        help="Checkpoint name.")
    parser.add_argument("--dataset_loc", type=int, default=100,
                        help="Localization dataset id (default: 100).")
    parser.add_argument("--dataset_seg", type=int, default=101,
                        help="Fine segmentation dataset id (default: 101).")
    parser.add_argument("--margin", type=int, default=5,
                        help="Bounding box margin in voxels (default: 5).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda / cpu / mps.")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep the temporary folder for debugging.")
    args = parser.parse_args()

    input_folder = Path(args.i)
    output_folder = Path(args.o)

    # folds: convert numeric strings to int, keep 'all' as is
    folds = tuple(int(x) if x.isdigit() else x for x in args.f)

    device = torch.device(args.device)

    output_folder.mkdir(parents=True, exist_ok=True)
    tmp = output_folder / "tmp"
    pcmra_dir = tmp / "pcmra"
    pcmra_seg_dir = tmp / "pcmra_seg"
    cropped_dir = tmp / "cropped"
    cropped_seg_dir = tmp / "cropped_seg"
    for d in (pcmra_dir, pcmra_seg_dir, cropped_dir, cropped_seg_dir):
        d.mkdir(parents=True, exist_ok=True)

    cases = find_cases(input_folder)
    if not cases:
        raise RuntimeError(f"No *_0000.nii.gz files found in {input_folder}")

    print(f"[1/6] Creating PCMRA images for {len(cases)} case(s)...")
    ref_affines = {}      # affine/header of the 3D reference space per case
    for case in cases:
        mag_path = input_folder / f"{case}_0000.nii.gz"
        vel_path = input_folder / f"{case}_0001.nii.gz"

        mag, affine, header = load_nifti(mag_path)
        vel, affine_vel, header_vel = load_nifti(vel_path)

        # --- geometry sanity checks -------------------------------------- #
        check_input_geometry(affine.copy(), header.copy(), case, "magnitude (_0000)")
        check_input_geometry(affine_vel.copy(), header_vel.copy(), case, "velocity (_0001)")
        if not np.allclose(affine, affine_vel, atol=1e-4):
            _warn(f"[{case}] Affines of _0000 and _0001 differ; "
                  f"channels may not be spatially aligned.")

        pcmra = create_pcmra(mag, vel)

        # Build a 3D reference header/affine for the (temporally averaged) space
        header3d = header.copy()
        header3d.set_data_shape(pcmra.shape)

        nib.save(
            nib.Nifti1Image(pcmra, affine, header3d),
            str(pcmra_dir / f"{case}_0000.nii.gz"),
        )
        ref_affines[case] = (affine, header3d)

    print(f"[2/6] Segmenting PCMRA with Dataset{args.dataset_loc:03d}...")
    predictor_loc = build_predictor(
        args.dataset_loc, args.c_loc, folds, args.chk,
        None, device, args.p, args.tr_loc,
    )
    predictor_loc.predict_from_files(
        str(pcmra_dir),
        str(pcmra_seg_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    del predictor_loc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[3/6] Cropping original images to PCMRA bounding box...")
    bboxes = {}
    orig_shapes = {}
    n_channels = 2  # _0000 and _0001

    for case in cases:
        seg_path = pcmra_seg_dir / f"{case}.nii.gz"
        seg, _, _ = load_nifti(seg_path)
        bbox = bounding_box(seg, margin=args.margin)
        bboxes[case] = bbox

        for ch in range(n_channels):
            in_path = input_folder / f"{case}_{ch:04d}.nii.gz"
            data, affine, header = load_nifti(in_path)
            orig_shapes[(case, ch)] = data.shape

            cropped = data[bbox]

            # adjust affine for the crop origin
            new_affine = affine.copy()
            origin_shift = np.array([s.start for s in bbox[:3]])
            new_affine[:3, 3] += affine[:3, :3] @ origin_shift

            hdr = header.copy()
            hdr.set_data_shape(cropped.shape)
            nib.save(
                nib.Nifti1Image(cropped.astype(np.float32), new_affine, hdr),
                str(cropped_dir / f"{case}_{ch:04d}.nii.gz"),
            )

    print(f"[4/6] Segmenting cropped images with Dataset{args.dataset_seg:03d}...")
    predictor_seg = build_predictor(
        args.dataset_seg, args.c_seg, folds, args.chk,
        None, device, args.p, args.tr_seg,
    )
    predictor_seg.predict_from_files(
        str(cropped_dir),
        str(cropped_seg_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    del predictor_seg
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[5/6] Un-cropping segmentations to original space...")
    for case in cases:
        seg_cropped, _, hdr_cropped = load_nifti(cropped_seg_dir / f"{case}.nii.gz")

        # Reference: spatial shape of channel 0 (first 3 dims)
        ref_shape = orig_shapes[(case, 0)][:4]
        full = np.zeros(ref_shape, dtype=np.uint8)

        bbox = bboxes[case]
        full[bbox[0], bbox[1], bbox[2],:] = seg_cropped.astype(np.uint8)

        affine, _ = ref_affines[case]
        out_hdr = hdr_cropped.copy()
        out_hdr.set_data_shape(full.shape)
        out_hdr.set_data_dtype(np.uint8)

        nib.save(
            nib.Nifti1Image(full, affine, out_hdr),
            str(output_folder / f"{case}.nii.gz"),
        )

    # 7. Cleanup ------------------------------------------------------------ #
    print("[6/6] Done.")
    if not args.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()