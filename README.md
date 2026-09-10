# Image Reconstruction from Radial Sampling of Frequency Data

Reconstructing medical images from non uniform samples (radial sampling) frequency domain data and comparing Inverse Non Uniform Fourier Transform (INUFT) and Frequency Domain Interpolation (FDI) reconstruction methods across different sampling ratios.

The setup is using a simplified method for accelerated image acquisition in medical imaging (ex: undersampled MRI k space). We only use a fraction of the freq coefficients and try recovering the image from the incomplete measurement.

## Problem Statement

Given an input image, obtain its 2d Fourier transform, retain the coeffs that fall on a set of radial spokes through the k space center and perform image reconstruction from the partial spectrum. 

Lower sampling ratio implies faster acquisiton, but less data to reconstruct from.

## Methodology

Radial Undersampling: Golden angle spokes added through frequency domain center until target fraction of coeffs is reached. A small centre is fully sampled, which keeps low freq energy and guarantees non degenerate sample set for interpolation.

INUFT: The unmeasured coeffs are left at zero and the complex part is inverted directly, this is considered as the baseline.

FDI: The missing coeffs are estimated by using scattered 2d linear interpolation of the measured coeffs, interpolating both real and imaginary parts. Measured coeffes are kept exact

Metrics: Reconstructed images are scored against ground truth using PSNR (higher PSNR means good reconstruction) and MIE (lower MIE means good reconstruction), these metrics are averaged over the different sampling ratios.


## Results

Averaged over 14 images of different chest diseases.

| Sampling Ratio | INUFT PSNR (dB) | INUFT MIE | FDI PSNR (dB) | FDI MIE |
|:---:|:---:|:---:|:---:|:---:|
|0.1|27.00|0.0323|24.56|0.0405|
|0.5|37.51|0.0098|34.45|0.0120|
|0.8|39.93|0.0076|37.49|0.0090|


## Observations
- Reconstruction quality improves monotonically based on the sampling ratio in both methods.
- INUFT outperforms FDI at each sampling ratio

## Usage

``` bash
pip install numpy scipy opencv-python matplotlib
```

1. Download the ChestXRay-14 dataset (or any set of grayscale images) and point them to 'img_dir' at the location where they're placed.
2. Run
```bash 
python signal_recon.py
```
This runs a batch evaluations, prints the PSNR/MIE tables and displays the images for each class at different ratios. 

## Repository Structure

```
`
|---singal_recon.py #sampling, reconstruction, evaluation and display of images
|---README.md
|---.gitignore #excludes the dataset
```

## Notes
- Radial spokes overlap near to the centre, so coverage gains diminish as spokes are added.

## Future Work
- Add SSIM as a new metric


