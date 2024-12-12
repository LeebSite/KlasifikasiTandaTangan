import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Path dataset lokal
dataset_path = 'Dataset'

# Fungsi preprocessing
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cropped = binary[10:118, 10:118]
    resized = cv2.resize(cropped, (128, 128))
    return resized, binary

# Fungsi ekstraksi fitur
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
    features.append(np.std(image))
    features.append(skew(image.flatten()))
    features.append(kurtosis(image.flatten()))

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

# Load dataset dan ekstraksi fitur
all_data = []
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(person_path, filename)
                resized, binary = preprocess_image(image_path)
                features = extract_features(binary)
                all_data.append([person, filename] + list(features))

# Define columns
columns = ["Person", "Filename"] + [f"Hist_{i}" for i in range(256)] + \
          ["Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation"] + \
          [f"Hu_{i}" for i in range(7)] + ["StdDev", "Skewness", "Kurtosis", "CenterX", "CenterY", "PixelDensity", "Eccentricity"]

df_features = pd.DataFrame(all_data, columns=columns)

# Remove columns with zero variance
df_features = df_features.loc[:, (df_features != 0).any(axis=0)]

# Normalisasi Data
X = df_features.iloc[:, 2:].values
y = df_features["Person"].factorize()[0]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Validasi Silang
scores = cross_val_score(knn, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.2f}")

# Evaluasi Model
y_pred = knn.predict(X_test)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred))

# Simpan Model dan Skala
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Fungsi Prediksi
def predict_new_image(model, scaler, img_path, label_map):
    resized, binary = preprocess_image(img_path)
    features = extract_features(binary)
    features = np.array(features[:model.n_features_in_]).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return label_map[prediction[0]]

# Capture Image
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak tersedia.")
        return None
    print("Tekan 's' untuk menangkap gambar, 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap gambar.")
            break
        cv2.imshow("Kamera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print("Gambar disimpan:", image_path)
            cap.release()
            cv2.destroyAllWindows()
            return image_path
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

# Jalankan prediksi
image_path = capture_image_from_camera()
if image_path:
    knn_model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_map = {i: label for i, label in enumerate(df_features["Person"].unique())}
    predicted_class = predict_new_image(knn_model, scaler, image_path, label_map)
    print(f"Tanda tangan milik: {predicted_class}")
