# import modul yang dibutuhkan
import cv2
from matplotlib import pyplot as plt

# import gambar yang ingin dibaca
image= cv2.imread('image/image.jpg')

# Convert BGR (OpenCV default) to RGB for proper color representation in the plot
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a figure for the histogram
plt.figure()

# Define colors for the channels
color = ('r', 'g', 'b')

# Loop through the RGB channels
for i, col in enumerate(color):
    # Calculate the histogram for each channel
    histr = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    # Plot the histogram
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
# Tampilkan plot (opsional)
plt.show()

# simpan hasil histogram ke nama dan ekstensi yang diinginkan
plt.savefig('image/rgb_histo.jpg', dpi=100)

# Close the plot to avoid display issues
plt.close()