import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import pywt
import os
import pandas as pd
from skimage.io import imread

# Define folder path
folder_path = "../archive/train/"

# Initialize lists for storing feature data
fourier_features = []
wavelet_features_list = []
counter = 0
# Iterate through all images in the folder
for person in os.listdir(folder_path):
    person_path = os.path.join(folder_path,person)
    
    for filename in os.listdir(person_path):
        if filename.endswith(".jpg"):  # Process only image files
            img_path = os.path.join(person_path,filename)
            img = imread(img_path)
            img = rgb2gray(img)
            img = img.astype(np.float32)

            # -------------------------------------------------------------
            # Compute the 2D Fourier Transform (FFT)
            # -------------------------------------------------------------
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)  # Shift zero frequency to center
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)  # Log-scale for better visualization

            # Extract FFT Features
            fft_mean = np.mean(magnitude_spectrum)
            fft_variance = np.var(magnitude_spectrum)
            fft_energy = np.sum(np.abs(magnitude_spectrum) ** 2)

            fourier_features.append({
                'Person' : person,
                "Image": filename,
                "FFT Mean": fft_mean,
                "FFT Variance": fft_variance,
                "FFT Energy": fft_energy
            })

        # -------------------------------------------------------------
        # Compute a 2D Wavelet Transform (Haar wavelet)
        # -------------------------------------------------------------
        coeffs2 = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs2

        wavelet_features_list.append({
            'Person' : person,
            "Image": filename,
            "cA Mean": np.mean(cA), "cA Variance": np.var(cA), "cA Energy": np.sum(cA ** 2),
            "cH Mean": np.mean(cH), "cH Variance": np.var(cH), "cH Energy": np.sum(cH ** 2),
            "cV Mean": np.mean(cV), "cV Variance": np.var(cV), "cV Energy": np.sum(cV ** 2),
            "cD Mean": np.mean(cD), "cD Variance": np.var(cD), "cD Energy": np.sum(cD ** 2)
        })
    print(f'{person} has been extracted')
    counter +=1
    print(counter,'/1245 completed.')


# Convert lists to DataFrames
fourier_df = pd.DataFrame(fourier_features)
wavelet_df = pd.DataFrame(wavelet_features_list)

# Save to Excel
with pd.ExcelWriter("features_fourier.xlsx") as writer:
    fourier_df.to_excel(writer, sheet_name="Fourier Features", index=False)
with pd.ExcelWriter("features_wavelet.xlsx") as writer:
    wavelet_df.to_excel(writer, sheet_name="Wavelet Features", index=False)

print("Feature extraction completed. Data saved to features_fourier.xlsx and features_wavelet.xlsx")
