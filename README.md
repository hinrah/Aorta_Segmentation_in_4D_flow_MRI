# Aorta Segmentation in 4D flow MRI
This repository allows you to perform an automatic segmentation of the aorta in 4D flow MRI using a 4D U-Net.
If you have any questions or remarks, please contact: hinrich.rahlfs@dhzc-charite.de

# Installation
## 1. Basics
- Use Python 3.10 or newer (Code was only tested against 3.10)
- Linux is the primary target. If you use windows, you will need to adapt the provided bash commands accordingly.
- GPU is strongly recommended 4D inference. Nvidia GPUs are the primary architecture. If you use CPUs or Apple mps, you will need to adapt the pytorch installation accordingly.

## 2. Get the Repository

```bash
git clone https://github.com/hinrah/Aorta_Segmentation_in_4D_flow_MRI.git
```

## 3. Install PyTorch first

Install PyTorch for your hardware before installing 4D nnunet:

<https://pytorch.org/get-started/locally/>

Choose `cuda` for NVIDIA GPUs.

Do not install the requirements before PyTorch is in place.

## 4. Install Repository requirements
```bash
pip install -r requirements.txt
```
## 5. Get the trained Models
The trained can be found here and will be available after acceptance of the article:
https://zenodo.org/records/22255537

Extract the zip file to "/path/to/repo/nnUNet_results" so that it looks like:

 ```
 /path/to/repo/nnUNet_results/
     Dataset100_PCMRA_Dataset/
        ...
     Dataset101_4D_flow_Aorta/
        ...
```

## 5. Set environment variables

### Linux 

```bash
export nnUNet_raw="/path/to/repo/nnUNet_raw"
export nnUNet_preprocessed="/path/to/repo/nnUNet_preprocessed"
export nnUNet_results="/path/to/repo/nnUNet_results"
```


# Inference
## 1. Prepare Files

The Networks expects the MRI magnitude (_0000) and the velocity magnitude (_0001) of the 4D flow MRI as Nifti files. These files need to be 4D with the temporal dimension beeing the last. The affine of the Nifti needs to be set as well as the correct voxel spacings.
The files need to be saved in an input folder with this structure:


```
input_folder/
   caseid1_0000.nii.gz
   caseid1_0001.nii.gz
   caseid2_0000.nii.gz
   caseid2_0001.nii.gz
```

## 2. Inference
To run inference: 
```bash
python run_inference.py -i /path/to/input -o /path/to/output/folder
```

If you want to see the results of the ROI cropping, pass --keep-temp as parameter.