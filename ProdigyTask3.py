import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow as tf

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available, using CPU.")

# Set paths
csv_path = 'C:\\Users\\Tharun Raman\\OneDrive\\Documents\\GitHub\\PRODIGY_ML_03\\dataset\\sampleSubmission.csv'
image_dir = 'C:\\Users\\Tharun Raman\\OneDrive\\Documents\\GitHub\\PRODIGY_ML_03\\dataset\\train\\train'

# Load the CSV data
data = pd.read_csv(csv_path)

# Prepare images and labels
images = []
labels = []

for index, row in data.iterrows():
    img_id = row['id']
    img_label = row['label']
    img_filename = f'cat.{img_id}.jpg' if img_label == 0 else f'dog.{img_id}.jpg'
    img_path = os.path.join(image_dir, img_filename)

    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(64, 64))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(img_label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Print shapes of images and labels
print(f'Total images loaded: {len(images)}')
print(f'Total labels collected: {len(labels)}')

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.25, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('cat_dog_classifier.keras')

# Classification code
model = load_model('cat_dog_classifier.keras')
test_image_dir = 'C:\\Users\\Tharun Raman\\OneDrive\\Documents\\GitHub\\PRODIGY_ML_03\\dataset\\test1'

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

results = []

for i in range(1, 12501):
    img_filename = os.path.join(test_image_dir, f'{i}.jpg')
    if os.path.exists(img_filename):
        processed_image = load_and_preprocess_image(img_filename)
        prediction = model.predict(processed_image)
        label = 'Dog' if prediction[0][0] >= 0.5 else 'Cat'
        results.append((i, label))
        print(f'Image ID: {i}, Predicted Label: {label}')
    else:
        print(f'Image ID: {i} not found.')

results_df = pd.DataFrame(results, columns=['Image_ID', 'Predicted_Label'])
results_df.to_csv('classification_results.csv', index=False)

print("Classification complete. Results saved to 'classification_results.csv'.")
