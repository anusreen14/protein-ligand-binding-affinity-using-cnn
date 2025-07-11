# train_cnn.py — Training script for 3D CNN on voxelized binding affinity data
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load voxelized data
with open("data/train_grids.pkl", "rb") as f:
    X = pickle.load(f)
with open("data/train_labels.pkl", "rb") as f:
    y = pickle.load(f)

print(f"Loaded data: {X.shape[0]} samples, voxel shape: {X.shape[1:]}\n")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Build 3D CNN model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv3D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv3D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))  # regression output

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Initialize and train model
model = build_model(input_shape=X.shape[1:])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    verbose=1
)

# Evaluate
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"\n✅ Evaluation on validation set:\nRMSE: {rmse:.4f}\n")

# Save the model
model.save("models/cnn_binding_affinity.keras")
print("\n✅ Model saved to models/cnn_binding_affinity.keras")
