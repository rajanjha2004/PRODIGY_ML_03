import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def load_images_smaller(folder_path, label):
    images = []
    labels = []
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (32, 32))
            images.append(image)
            labels.append(label)
    return images, labels

dataset_path = './dogs_vs_cats/train'
cat_images_small, cat_labels = load_images_smaller(os.path.join(dataset_path, 'cats'), 0)
dog_images_small, dog_labels = load_images_smaller(os.path.join(dataset_path, 'dogs'), 1)

X_small = np.array(cat_images_small + dog_images_small)
y_small = np.array(cat_labels + dog_labels)
X_small = X_small / 255.0
X_small = X_small.reshape(len(X_small), -1)
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X_small, y_small, test_size=0.2, random_state=42)

print("Training data shape:", X_train_small.shape)
print("Testing data shape:", X_test_small.shape)

scaler = StandardScaler()
X_train_small_scaled = scaler.fit_transform(X_train_small)
X_test_small_scaled = scaler.transform(X_test_small)

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_small_scaled, y_train_small)

y_pred_small = svm_model.predict(X_test_small_scaled)

accuracy_small = accuracy_score(y_test_small, y_pred_small)
print(f"Accuracy on smaller dataset: {accuracy_small * 100:.2f}%")

cm_small = confusion_matrix(y_test_small, y_pred_small)
print("Confusion Matrix:\n", cm_small)

sns.heatmap(cm_small, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Cats vs Dogs Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image.reshape(1, -1)
    image = image / 255.0
    image = scaler.transform(image)
    prediction = svm_model.predict(image)
    if prediction == 0:
        return 'Cat'
    else:
        return 'Dog'

test_image_path = './dogs_vs_cats/test/cats/cat.10.jpg'
print(predict_image(test_image_path))
