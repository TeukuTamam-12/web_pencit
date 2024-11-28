from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

app = Flask(__name__)

# Path untuk upload gambar
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path untuk model DNN
MODEL_PATH_FACE = 'static/models/res10_300x300_ssd_iter_140000.caffemodel'
CONFIG_PATH_FACE = 'static/models/deploy.prototxt'

# Load DNN model
net_face = cv2.dnn.readNetFromCaffe(CONFIG_PATH_FACE, MODEL_PATH_FACE)

# Halaman utama untuk upload gambar
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk memproses upload dan generate histogram, deteksi wajah, blur wajah, deteksi tepi, atau ganti background
@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    background_file = request.files.get('background')
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if background_file and background_file.filename != '':
            background_filename = secure_filename(background_file.filename)
            background_filepath = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)
            background_file.save(background_filepath)
        else:
            background_filepath = None

        process_type = request.form.get('process_type')

        # Handle chain code separately
        if process_type == 'chain_code':
            chain_code_result = calculate_chain_code_from_image(cv2.imread(filepath))
            if chain_code_result:
                chain_code_str = '  '.join(map(str, chain_code_result))
            else:
                chain_code_str = "Tidak ada kontur yang ditemukan."
            return render_template('result.html', 
                                image_file=filename, 
                                process_type=process_type, 
                                chain_code=chain_code_str)

        # Handle other image processing operations
        if process_type == 'grayscale':
            generate_grayscale_histogram(filepath)
        elif process_type == 'rgb':
            generate_rgb_histogram(filepath)
        elif process_type == 'face_detection':
            detect_faces_dnn(filepath)
        elif process_type == 'blur_faces':
            blur_faces_dnn(filepath, 'static/blurred_faces.jpg')
        elif process_type == 'edge_detection':
            edge_detection_dnn(filepath, 'static/edges_detected.jpg')
        elif process_type == 'restore_image':
            restore_image(filepath, 'static/restored_image.jpg')
        elif process_type == 'enhance_image':
            enhance_image(filepath, 'static/enhanced_image.jpg')
        elif process_type == 'dilation_image':
            dilation(filepath, 'static/dilated_image.jpg')
        elif process_type == 'erosion_image':
            erosion(filepath, 'static/eroded_image.jpg')
        elif process_type == 'opening_image':
            apply_opening(filepath, 'static/opening_out.jpg')
        elif process_type == 'closing_image':
            apply_closing(filepath, 'static/closing_out.jpg')
        elif process_type == 'morpho_image':
            detect_edges_morphological_gradient(filepath, 'static/morpho_out.jpg')
        elif process_type == 'nearest_neighbor_image':
            nearest_neighbor_interpolation(filepath, 'static/nearest_neighbor_out.jpg')
        elif process_type == 'bilinear_image':
            bilinear_interpolation(filepath, 'static/bilinear_out.jpg')
        elif process_type == 'bicubic_image':
            bicubic_interpolation(filepath, 'static/bicubic_out.jpg')
        elif process_type == 'change_background' and background_filepath:
            rect = (50, 50, 450, 290)
            change_background_grabcut(filepath, background_filepath, rect, 'static/background_changed.jpg')

        return render_template('result.html', 
                             image_file=filename, 
                             process_type=process_type)

    return redirect(request.url)
           
# Fungsi untuk deteksi tepi (Edge Detection)
def edge_detection_dnn(image_path, output_path):
    # Load the pre-trained model for HED (Holistically-Nested Edge Detection)
    model_path = "static/models/hed_pretrained_bsds.caffemodel"
    config_path = "static/models/deploy1.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)

    # Pass the blob through the network and get the edge map
    net.setInput(blob)
    edge_map = net.forward()

    # Reshape the edge map to the original image size
    edge_map = edge_map[0, 0]
    edge_map = cv2.resize(edge_map, (w, h))

    # Normalize the edge map for proper visualization
    edge_map = (255 * edge_map).astype("uint8")

    # Save the edge map to the output path
    cv2.imwrite(output_path, edge_map)
    print(f"Edge detection image saved as {output_path}")

# Fungsi untuk generate histogram grayscale
def generate_grayscale_histogram(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Membuat histogram grayscale
    histr = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Plot histogram dan simpan di folder static
    plt.figure()
    plt.plot(histr)
    plt.title('Histogram Grayscale')
    plt.savefig('static/gray_histo.jpg', dpi=100)
    plt.close()

    # Simpan gambar grayscale ke folder static
    cv2.imwrite('static/gray_image.jpg', gray_image)

# Fungsi untuk generate histogram RGB
def generate_rgb_histogram(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Membuat histogram RGB
    plt.figure()
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.title('Histogram RGB')
    plt.savefig('static/rgb_histo.jpg', dpi=100)
    plt.close()

# Fungsi untuk deteksi wajah menggunakan model DNN
def detect_faces_dnn(image_path):
    # Load the image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Set the input to the network
    net_face.setInput(blob)

    # Perform forward pass to get face detections
    detections = net_face.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is greater than a threshold
        if confidence > 0.5:
            # Compute the coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Draw the bounding box around the detected face
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

    # Simpan gambar hasil deteksi wajah di folder static
    cv2.imwrite('static/face_detected.jpg', img)

# Fungsi untuk blur wajah menggunakan model DNN
def blur_faces_dnn(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Prepare the image for the neural network
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network and get the detections
    net_face.setInput(blob)
    detections = net_face.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.5:
            # Compute the coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within the dimensions of the image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Extract the face region of interest (ROI)
            face = image[startY:endY, startX:endX]

            # Apply Gaussian blur to the face
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)

            # Replace the original face with the blurred face in the image
            image[startY:endY, startX:endX] = blurred_face

    # Save the output image with blurred faces
    cv2.imwrite(output_path, image)
    print(f"Blurred face image saved as {output_path}")

def change_background_grabcut(image_path, background_path, rect, output_path):
    # Load input image and background image
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)

    # Resize background to match input image size
    image_height, image_width = image.shape[:2]
    background = cv2.resize(background, (image_width, image_height))

    # Initialize mask, bgdModel, and fgdModel for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify mask to extract the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image * mask2[:, :, np.newaxis]

    # Extract the background where the mask is 0
    background_part = background * (1 - mask2[:, :, np.newaxis])

    # Combine the foreground and new background
    result = cv2.add(foreground, background_part)

    # Save the resulting image
    cv2.imwrite(output_path, result)

def restore_image(image_path, output_path, kernel_size=5):
    # Baca citra menggunakan OpenCV
    image = cv2.imread(image_path)
    
    # Dapatkan dimensi citra
    (h, w) = image.shape[:2]
    
    # Konversi ke grayscale jika citra berwarna
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Terapkan filter median untuk mengurangi noise
    restored = cv2.medianBlur(gray, kernel_size)
    
    # Normalisasi citra hasil restorasi untuk visualisasi yang lebih baik
    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)
    
    # Simpan citra hasil restorasi
    cv2.imwrite(output_path, restored)
    print(f"Citra hasil restorasi disimpan sebagai {output_path}")

def enhance_image(image_path, output_path, kernel_size=5, sharpening_strength=1.0):
    # Baca citra sebagai grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Langkah 1: Deblurring menggunakan filter Wiener
    psf = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    deblurred = cv2.filter2D(image, -1, psf)
    noise = 10
    deblurred = cv2.addWeighted(image, 1.0 + sharpening_strength, deblurred, -sharpening_strength, noise)
    
    # Langkah 2: Peningkatan ketajaman menggunakan Unsharp Masking
    gaussian = cv2.GaussianBlur(deblurred, (0, 0), 2.0)
    sharpened = cv2.addWeighted(deblurred, 1 + sharpening_strength, gaussian, -sharpening_strength, 0)
    
    # Langkah 3: Peningkatan kontras menggunakan CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharpened)
    
    # Langkah 4: Reduksi noise menggunakan Non-local Means Denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Normalisasi hasil akhir
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    # Simpan citra hasil enhancement
    cv2.imwrite(output_path, enhanced)
    print(f"Citra hasil enhancement disimpan sebagai {output_path}")

def dilation(image_path, output_path):
    # Memuat gambar dalam format grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Terapkan threshold untuk mengubah gambar menjadi biner
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Definisikan kernel untuk operasi dilation (misalnya, matriks 5x5 ones)
    kernel = np.ones((5, 5), np.uint8)

    # Lakukan operasi dilation
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Simpan gambar hasil erosi
    cv2.imwrite(output_path, dilated_image)
    print(f"Blurred face image saved as {output_path}")

def erosion(image_path, output_path):
    # Memuat gambar dalam format grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Terapkan threshold untuk mengubah gambar menjadi biner
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Definisikan kernel untuk operasi erosion (misalnya, matriks 5x5 ones)
    kernel = np.ones((5, 5), np.uint8)

    # Lakukan operasi erosion
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Simpan gambar hasil erosi
    cv2.imwrite(output_path, eroded_image)

def apply_opening(image_path, output_path, kernel_size=5):
    # Baca gambar sebagai grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarisasi gambar (thresholding)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Definisikan kernel untuk operasi opening
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Terapkan operasi opening
    opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Simpan hasil
    cv2.imwrite(output_path, opening_image)
    print(f"Gambar hasil opening disimpan sebagai {output_path}")

def apply_closing(image_path, output_path, kernel_size=5):
    # Baca gambar sebagai grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarisasi gambar (thresholding)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Definisikan kernel untuk operasi closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Terapkan operasi closing
    closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Simpan hasil
    cv2.imwrite(output_path, closing_image)
    print(f"Gambar hasil closing disimpan sebagai {output_path}")


def detect_edges_morphological_gradient(image_path, output_path, kernel_size=5):
    # Baca citra sebagai grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Langkah 1: Definisikan kernel (elemen struktural) untuk operasi morfologi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Langkah 2: Terapkan operasi dilasi
    dilated = cv2.dilate(image, kernel)

    # Langkah 3: Terapkan operasi erosi
    eroded = cv2.erode(image, kernel)

    # Langkah 4: Hitung morphological gradient (selisih antara dilasi dan erosi)
    gradient = cv2.subtract(dilated, eroded)

    # Simpan citra hasil deteksi tepi
    cv2.imwrite(output_path, gradient)
    print(f"Deteksi tepi dengan morphological gradient disimpan sebagai {output_path}")

def nearest_neighbor_interpolation(image_path, output_path, scale_factor=2):
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Hitung dimensi baru
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    # Lakukan interpolasi nearest neighbor
    interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Simpan hasil
    cv2.imwrite(output_path, interpolated_image)
    print(f"Gambar hasil interpolasi nearest neighbor disimpan sebagai {output_path}")

def bilinear_interpolation(image_path, output_path, scale_factor=5):
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Hitung dimensi baru
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    # Lakukan interpolasi bilinear
    interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Simpan hasil
    cv2.imwrite(output_path, interpolated_image)
    print(f"Gambar hasil interpolasi bilinear disimpan sebagai {output_path}")

def bicubic_interpolation(image_path, output_path, scale_factor=10):
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Hitung dimensi baru
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    # Lakukan interpolasi bicubic
    interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Simpan hasil
    cv2.imwrite(output_path, interpolated_image)
    print(f"Gambar hasil interpolasi bicubic disimpan sebagai {output_path}")


# Code list khusus dengan urutan berbeda
codeList = [5, 6, 7, 4, -1, 0, 3, 2, 1]  # Dengan nilai tidak valid (-1)


def getChainCode(dx, dy):
    hashKey = (3 * dy + dx + 4) % len(codeList)
    chainCode = codeList[hashKey]
    if chainCode < 0:
        chainCode += len(codeList)
    return chainCode

def generate_chain_code(ListOfPoints):
    chainCode = []
    for i in range(len(ListOfPoints) - 1):
        a = ListOfPoints[i]
        b = ListOfPoints[i + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        chainCode.append(getChainCode(dx, dy))
    return chainCode

def calculate_chain_code_from_image(image, visualize=False):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better binarization
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours with more detail (no approximation)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Select the largest contour (assumes it's the object of interest)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Optional: Visualize contours
        if visualize:
            contour_image = image.copy()
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Contours", contour_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Generate chain code from the largest contour
        if largest_contour.size > 0:
            chain_code = generate_chain_code([point[0] for point in largest_contour])
            return chain_code
    return None

if __name__ == "__main__":
    app.run(debug=True)