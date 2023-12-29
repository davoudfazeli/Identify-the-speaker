import os
import librosa
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


notes_path = './recordings/'
total_feature_dimensions = 39  # Update with the actual size of your feature vector

# Preallocate arrays for faster concatenation
mfcc_data = np.empty((0, 13))
delta_data = np.empty((0, 13))
double_delta_data = np.empty((0, 13))

# Get the total number of files to process
total_files = sum(1 for file in os.listdir(notes_path) if file.endswith(".wav"))
label = []
# Initialize tqdm with the total number of files
for root, dirs, files in os.walk(notes_path):
    for file in tqdm(files, total=total_files, desc='Processing audio files'):
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            y, sr = librosa.load(file_path)
            if file_path.find('jackson')!=-1:
                label.append(1)
            elif file_path.find('theo')!=-1:
                label.append(2)
            elif file_path.find('nicolas')!=-1:
                label.append(3)
            # Compute MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_data = np.vstack((mfcc_data, np.mean(mfccs, axis=1)))

            # Compute delta and double delta MFCC features
            delta_mfccs = librosa.feature.delta(mfccs)
            double_delta_mfccs = librosa.feature.delta(delta_mfccs)

            # Concatenate features
            delta_data = np.vstack((delta_data, np.mean(delta_mfccs, axis=1)))
            double_delta_data = np.vstack((double_delta_data, np.mean(double_delta_mfccs, axis=1)))

# Combine all features into a single array
data = np.hstack((mfcc_data, delta_data, double_delta_data))
label = np.array(label)
x=0

x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=50, test_size=0.2)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

# Normalize input data using MinMaxScaler
scaler_x = MinMaxScaler()
x_test_scaled = scaler_x.fit_transform(x_test)
x_train_scaled = scaler_x.fit_transform(x_train)


model = Sequential([
    Dense(128, activation='relu', input_shape=(total_feature_dimensions,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # Use softmax for multiclass classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Add learning rate scheduling
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

output = model.fit(x_train_scaled, y_train_encoded, epochs=100, validation_split=0.2, callbacks=[early_stopping, lr_scheduler], verbose=0)


y_pred = model.predict(x_test_scaled)
y_pred = np.argmax(y_pred,axis = 1)+1
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)


plt.plot(output.history['loss'], label='Training Loss')
plt.plot(output.history['val_loss'], label='Validation Loss')
plt.title(f'Train Results, RMSE = {rmse:.3f}')
plt.legend()
plt.savefig("Results.png")
plt.show()

