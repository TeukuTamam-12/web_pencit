# import modul yang dibutuhkan
import cv2
from matplotlib import pyplot as plt

# import gambar yang ingin dibaca
image= cv2.imread('image/image.jpg')

# mengubah gambar ke grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a figure for the histogram
plt.figure()

# membuat histogram dari gambar grayscale
# mengkalkulasikan warna dari gambar grayscale
histr = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# hasil kalkulasi menjadi plot
plt.plot(histr)

# Tampilkan plot (opsional)
plt.show()

# simpan hasil histogram ke nama dan ekstensi yang diinginkan dan menambahkan format gambar .jpg
plt.savefig(f'image/gray_histo.jpg', dpi=100)

# Close the plot to avoid display issues
plt.close()

# simpan gambar yang telah dijadikan grayscale
cv2.imwrite(f'image/gray_image.jpg', gray_image)
