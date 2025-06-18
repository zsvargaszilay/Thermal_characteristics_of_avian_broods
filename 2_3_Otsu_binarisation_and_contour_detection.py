import os
import numpy as np
import cv2
import matplotlib.pyplot as plt # for visualization

# Input: A temperature matrix from a thermal camera in CSV format
# Output: A CSV file containing the filtered temperature matrix for one thermal image
# The applied mask was generated using Otsu's thresholding technique
# and contour detection.

input_path = "...:/.../csv/"
output_path = "...:/.../otsu_and_konturdet_matrix_csv_with_transf/"

LOW_P  = 5   # lower percentile
HIGH_P = 95  # upper percentile

for filename in os.listdir(input_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_path, filename)
        #file_path = "d:/Work/Pannon_Uni_work/csv/Et19_06.07_feszek_IR004044.csv"
        temperature_matrix = np.genfromtxt(file_path, delimiter='\t', skip_header=1, invalid_raise=False)

        # Data transformation using percentiles
        # Extract non-NaN pixel values
        valid_pixels = temperature_matrix[~np.isnan(temperature_matrix)]

        # Determine percentiles
        p_low = np.percentile(valid_pixels, LOW_P)
        p_high = np.percentile(valid_pixels, HIGH_P)

        # Clipping to the selected range
        clipped = np.clip(temperature_matrix, p_low, p_high)

        # Normalization to 0-255
        normalized_matrix = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

        # Normalization to 0-255 (previous version without transformation)
        # min_temp = np.nanmin(temperature_matrix)
        # max_temp = np.nanmax(temperature_matrix)
        # normalized_matrix = ((temperature_matrix - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)

        # # Display histogram of the normalized image
        # plt.figure()
        # plt.title("Histogram of Normalized Matrix")
        # plt.hist(normalized_matrix.ravel(), bins=256, range=[0, 256])
        # plt.show()

        # Otsu binarization
        otsu_thresh, binary_threshold = cv2.threshold(normalized_matrix, 0, 255,
                                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #Check: Print threshold value
        #print(f"Otsu küszöbérték a(z) {filename} fájlhoz: {otsu_thresh}")

        # Contour detection
        contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Initialize mask
        mask = np.zeros_like(binary_threshold)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # Restore original matrix values within contours
        filtered_matrix = np.where(mask == 255, temperature_matrix, np.nan)

        # Filtering: keep values only where the binary image is white (255) (no contour detection)
        #filtered_matrix = np.where(binary_threshold == 255, temperature_matrix, np.nan)

        # Display results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 4, 1)
        plt.title("Original Thermal Image")
        plt.imshow(temperature_matrix, cmap="inferno")
        plt.colorbar()

        plt.subplot(1, 4, 2)
        plt.title("Normalized Matrix")
        plt.imshow(normalized_matrix, cmap="gray")
        plt.colorbar()

        plt.subplot(1, 4, 3)
        plt.title("Otsu's Binarization")
        plt.imshow(binary_threshold, cmap="gray")

        plt.subplot(1, 4, 4)
        plt.title("Final")
        plt.imshow(filtered_matrix, cmap="gray")
        plt.show()

        # Save filtered matrix as CSV file with tab delimiter
        output_filename = os.path.splitext(filename)[0] + "_filtered.csv"
        output_file_path = os.path.join(output_path, output_filename)
        #np.savetxt(output_file_path, filtered_matrix, delimiter='\t', fmt='%.2f')

        print(f"Filtered file saved: {output_file_path}")



