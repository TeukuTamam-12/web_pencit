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
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    background_file = request.files.get('background')  # Optional background file
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Jika background file juga diupload, simpan di folder uploads
        if background_file and background_file.filename != '':
            background_filename = secure_filename(background_file.filename)
            background_filepath = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)
            background_file.save(background_filepath)
        else:
            background_filepath = None

        # Pilih tipe proses dari form (RGB, Grayscale, Deteksi Wajah, Blur Wajah, Deteksi Tepi, atau Ganti Background)
        process_type = request.form.get('process_type')
        if process_type == 'grayscale':
            generate_grayscale_histogram(filepath)
        elif process_type == 'rgb':
            generate_rgb_histogram(filepath)
        elif process_type == 'face_detection':
            detect_faces_dnn(filepath)
        elif process_type == 'blur_faces':
            output_path = 'static/blurred_faces.jpg'
            blur_faces_dnn(filepath, output_path)
        elif process_type == 'edge_detection':
            output_path = 'static/edges_detected.jpg'
            edge_detection_dnn(filepath, output_path)
        elif process_type == 'change_background' and background_filepath:
            output_path = 'static/background_changed.jpg'
            rect = (50, 50, 450, 290)  # Rect harus disesuaikan untuk setiap gambar
            change_background_grabcut(filepath, background_filepath, rect, output_path)
        elif process_type == 'restore_image':
            output_path = 'static/restored_image.jpg'
            restore_image(filepath, output_path)
        elif process_type == 'enhance_image':
            output_path = 'static/enhanced_image.jpg'
            enhance_image(filepath, output_path)

        # Redirect untuk menampilkan hasil
        return render_template('result.html', image_file=filename, process_type=process_type)
    
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
if __name__ == "__main__":
    app.run(debug=True)
