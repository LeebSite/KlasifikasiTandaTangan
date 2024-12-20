import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # Klasifikasi dengan KNN
from sklearn.metrics import classification_report # Laporan hasil klasifikasi
from sklearn.preprocessing import StandardScaler # Normalisasi data

# Fungsi untuk preprocessing gambar
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {img_path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(binary, (128, 128))
    return resized, binary

# Ekstraksi fitur
def extract_features(image):
    features = []
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist = hist / np.sum(hist)  # Normalisasi histogram
    features.extend(hist)
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features.append(graycoprops(glcm, prop).mean())
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(hu_moments)
    features.append(np.std(image))
    features.append(skew(image.flatten()))
    features.append(kurtosis(image.flatten()))
    return np.nan_to_num(features)

# Membaca dataset
def load_dataset(dataset_path):
    data = []
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(person_path, file)
                    try:
                        resized, binary = preprocess_image(img_path)
                        features = extract_features(resized)
                        data.append([person] + features.tolist())
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    return data

# Main function
def main():
    dataset_path = "Dataset"
    data = load_dataset(dataset_path)

    # Konversi ke DataFrame
    columns = ["Person"] + [f"Feature_{i}" for i in range(len(data[0]) - 1)]
    df = pd.DataFrame(data, columns=columns)

    # Pemisahan fitur dan label
    X = df.iloc[:, 1:].values
    y = df["Person"].factorize()[0]
    label_map = {i: label for i, label in enumerate(df["Person"].unique())}

    # Normalisasi fitur
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=df["Person"].unique()))

    # Prediksi gambar dari kamera
    print("Tekan 'C' untuk menangkap gambar.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Kamera", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            capture_path = "capture.png"
            cv2.imwrite(capture_path, frame)
            resized, binary = preprocess_image(capture_path)
            features = extract_features(resized)
            features = scaler.transform([features])
            pred_label = model.predict(features)[0]
            print(f"Tanda tangan diprediksi sebagai: {label_map[pred_label]}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
