import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, interp1d
import os


def compute_mie(orig, reconstruct, range_min = None, range_max = None):
    if range_min is None:
        range_min = np.min(orig)
    if range_max is None:
        range_max = np.max(orig)
    range_size = range_max - range_min

    integral_error = np.sum(np.abs(orig-reconstruct))

    return integral_error/ (orig.size * range_size)

def compute_psnr(orig, reconstruct, max_val=255):
  mse = np.mean((orig - reconstruct)**2)

  return 10 * np.log10((max_val ** 2) / mse)

def ift(sample_freq):
  return np.abs(np.fft.ifft2(np.fft.ifftshift(sample_freq)))

def fdi(sample_freq_domain, result):
  h, w = sample_freq_domain.shape

  mask1 = sample_freq_domain != 0
  y_coord, x_coord = np.nonzero(mask1)
  known_vals = sample_freq_domain[mask1]

  gridY, gridX = np.mgrid[0:h, 0:w]
  pts = np.column_stack((y_coord, x_coord))


  interp_real_part = griddata(pts, known_vals.real, (gridY, gridX),
                              method='linear', fill_value=0.0)
  interp_imag_part = griddata(pts, known_vals.imag, (gridY, gridX),
                              method='linear', fill_value=0.0)
  new_freq_domain = interp_real_part + 1j * interp_imag_part

  new_freq_domain[mask1] = known_vals

  reconstructed_img = np.abs(np.fft.ifft2(np.fft.ifftshift(new_freq_domain)))

  return reconstructed_img



def radialNonUniSampling(freq, sample_ratio):
  h, w  = freq.shape
  centre = np.array([h/2.0, w/2.0])
  max_radius = np.hypot(h, w)/2.0
  radii = np.arange(-max_radius, max_radius, 0.5)

  golden_angle = np.deg2rad(111.25)
  mask1 = np.zeros(freq.shape, dtype=bool)
  target = sample_ratio * freq.size
  cy, cx = int(centre[0]), int(centre[1])
  c = 3
  mask1[max(cy-c, 0):cy+c+1, max(cx-c, 0):cx+c+1] = True

  angle = 0.0

  for _ in range(int(4 * max(h, w))):
    ys = np.round(centre[0] + radii * np.sin(angle)).astype(int)
    xs = np.round(centre[1] + radii * np.cos(angle)).astype(int)

    good_px = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    mask1[ys[good_px], xs[good_px]] = True
    angle+=golden_angle

    if(mask1.sum() >= target):
      break

  sampled_freqs = np.zeros_like(freq)
  sampled_freqs[mask1] = freq[mask1]

  return sampled_freqs

img_dir = r"C:\Users\remel\Desktop\INUFT\img_dir\img_dir"

num_imgs = 14

img_files = os.listdir(img_dir)[:num_imgs]
sampling_ratio = [0.1, 0.5, 0.8]

mie_inuft_results = [[] for _ in sampling_ratio]
psnr_inuft_results = [[] for _ in sampling_ratio]
mie_fdi_results = [[] for _ in sampling_ratio]
psnr_fdi_results = [[] for _ in sampling_ratio]


for i, img in enumerate(img_files):
  img_path = os.path.join(img_dir, img)
  img_curr = cv2.imread(img_path)
  img_gray = np.mean(img_curr, axis=2)
  img_gray = cv2.resize(img_gray.astype(np.float32), (256, 256)).astype(np.float64)
  img_label = os.path.splitext(img)[0]

  f = np.fft.fft2(img_gray)
  fshift = np.fft.fftshift(f)

  sampled_fshift = []
  recon_inuft_gray_imgs = []
  recon_fdi_gray_imgs = []

  plt.figure(figsize=(10, 9))

  for j in range(len(sampling_ratio)):
    sampled_fshift.append(radialNonUniSampling(fshift, sampling_ratio[j]))
    recon_inuft_gray_imgs.append(ift(sampled_fshift[j]))

    op_shape = recon_inuft_gray_imgs[j].shape
    recon_fdi_gray_imgs.append(fdi(sampled_fshift[j], op_shape))

    mie_inuft_results[j].append(compute_mie(img_gray, recon_inuft_gray_imgs[j]))
    psnr_inuft_results[j].append(compute_psnr(img_gray, recon_inuft_gray_imgs[j]))

    mie_fdi_results[j].append(compute_mie(img_gray, recon_fdi_gray_imgs[j]))
    psnr_fdi_results[j].append(compute_psnr(img_gray, recon_fdi_gray_imgs[j]))

  #   plt.subplot(3, 3, j*3 + 1)
  #   plt.imshow(img_gray, cmap='gray')
  #   plt.title('Original --' + img_label)
  #   plt.axis('off')

  #   plt.subplot(3, 3, j*3 + 2)
  #   plt.imshow(recon_inuft_gray_imgs[j], cmap='gray')
  #   plt.title('INUFT --' + img_label +'@ sampling ratio ' + str(sampling_ratio[j]))
  #   plt.axis('off')

  #   plt.subplot(3, 3, j*3 + 3)
  #   plt.imshow(recon_fdi_gray_imgs[j], cmap='gray')
  #   plt.title('FDI--' + img_label+'@ sampling ratio ' + str(sampling_ratio[j]))
  #   plt.axis('off')

  # plt.tight_layout()
  # plt.show()

print("INUFT")

for i, sr in enumerate(sampling_ratio):
  print(f"sr={sr:.2f} PSNR {np.mean(psnr_inuft_results[i]):.2f} MIE: {np.mean(mie_inuft_results[i]):.4f}")

print("FDI")

for i, sr in enumerate(sampling_ratio):
  print(f"sr={sr:.2f} PSNR {np.mean(psnr_fdi_results[i]):.2f} MIE: {np.mean(mie_fdi_results[i]):.4f}")


for i in range(len(sampling_ratio)):
  print(sampling_ratio[i], np.count_nonzero(sampled_fshift[i]))