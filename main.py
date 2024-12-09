import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

# Fungsi Preprocessing
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Greyscale
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # Binarisasi
    cropped = img_gray[10:118, 10:118]  # Cropping (misal ambil area tengah)
    resized = cv2.resize(cropped, (128, 128))  # Resize ke 128x128
    return resized, binary

# Fungsi Ekstraksi Fitur
def extract_features(image):
    features = []
    
    # Histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    features.extend(hist)
    
    # GLCM Features
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features.append(graycoprops(glcm, prop).mean())
    
    # Hu Moments
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(hu_moments)
    
    # Additional Features
    features.append(np.std(image))  # Standard Deviation
    features.append(skew(image.flatten()))  # Skewness
    features.append(kurtosis(image.flatten()))  # Kurtosis
    
    # Centroid
    if moments['m00'] != 0:
        cX = moments['m10'] / moments['m00']
        cY = moments['m01'] / moments['m00']
    else:
        cX, cY = 0, 0
    features.extend([cX, cY])
    
    # Pixel Density
    pixel_density = np.sum(image) / (image.shape[0] * image.shape[1])
    features.append(pixel_density)
    
    # Eccentricity
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(max(contours, key=cv2.contourArea)) >= 5:
        cnt = max(contours, key=cv2.contourArea)
        _, (ma, MA), _ = cv2.fitEllipse(cnt)
        eccentricity = np.sqrt(1 - (ma ** 2 / MA ** 2))
    else:
        eccentricity = 0
    features.append(eccentricity)
    
    return np.nan_to_num(features)

# Model Pelatihan (Gunakan model yang telah Anda latih sebelumnya)
# Simulasi Dataframe Fitur
df_features = pd.read_excel('Signature_Features.xlsx')  # Load dataset yang sebelumnya disimpan
X = df_features.iloc[:, 2:].values  # Exclude Person and Filename
y = df_features["Person"].factorize()[0]  # Encode labels

# Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Label Map untuk prediksi
label_map = {i: label for i, label in enumerate(df_features["Person"].unique())}
feature_columns = df_features.columns[2:].tolist()  # Kolom fitur

# Fungsi Prediksi
def predict_new_image(model, img, label_map, feature_columns):
    resized, binary = preprocess_image(img)
    features = extract_features(binary)
    
    # Pilih fitur yang digunakan saat training
    features = [features[i] for i in range(len(features)) if feature_columns[i] in feature_columns]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return label_map[prediction[0]]

# Fungsi untuk Capture Image
def capture_and_predict():
    # Inisialisasi Kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak dapat diakses!")
        return

    print("Tekan 's' untuk mengambil gambar, atau 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat menangkap frame.")
            break

        # Tampilkan frame
        cv2.imshow("Capture Image", frame)

        # Tunggu input keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Jika tekan 's', simpan dan prediksi
            print("Gambar diambil!")
            predicted_label = predict_new_image(knn, frame, label_map, feature_columns)
            print(f"Prediksi: {predicted_label}")
            cv2.imwrite("captured_image.png", frame)
            break
        elif key == ord('q'):  # Keluar
            break

    cap.release()
    cv2.destroyAllWindows()

# Jalankan Capture dan Prediksi
capture_and_predict()
