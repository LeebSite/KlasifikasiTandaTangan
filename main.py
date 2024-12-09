import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from google.colab.patches import cv2_imshow
from google.colab import files

# Preprocessing function
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cropped = img[10:118, 10:118]  # Example crop area
    resized = cv2.resize(cropped, (128, 128))
    return resized, binary

# Feature extraction function
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

    return np.nan_to_num(features)  # Replace NaN values with 0

# Load dataset and extract features
dataset_path = '/content/drive/My Drive/Dataset'
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

# Create DataFrame
df_features = pd.DataFrame(all_data, columns=columns)

# Remove columns where all values are zero
df_features = df_features.loc[:, (df_features != 0).any(axis=0)]

# Save to Excel
output_path = '/content/drive/My Drive/Signature_Features.xlsx'
df_features.to_excel(output_path, index=False)
print(f"Features saved to {output_path}")

# Example prediction using KNN
X = df_features.iloc[:, 2:].values  # Exclude Person and Filename
y = df_features["Person"].factorize()[0]  # Encode labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict new image
def predict_new_image(model, image_path, label_map):
    resized, binary = preprocess_image(image_path)
    features = extract_features(binary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return label_map[prediction[0]]

# Define label map
label_map = {i: label for i, label in enumerate(df_features["Person"].unique())}

# Upload and predict
uploaded = files.upload()
for filename in uploaded.keys():
    predicted_label = predict_new_image(knn, filename, label_map)
    print(f"Predicted label for {filename}: {predicted_label}")
