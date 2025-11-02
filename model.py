"""
Emotion Detection Model Training Script
Uses FER2013 dataset with transfer learning on VGGFace/ResNet50
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class EmotionDetectionModel:
    def __init__(self, dataset_path='fer2013.csv'):
        """
        Initialize the emotion detection model
        
        Args:
            dataset_path: Path to FER2013 dataset CSV file
        """
        self.dataset_path = dataset_path
        self.img_size = 224  # ResNet50 input size
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.num_classes = len(self.emotion_labels)
        self.model = None
        
    def load_and_preprocess_data(self):
        """Load FER2013 dataset and preprocess images"""
        print("Loading FER2013 dataset...")
        
        # Read the CSV file
        df = pd.read_csv(self.dataset_path)
        
        # Extract pixels and labels
        pixels = df['pixels'].tolist()
        emotions = df['emotion'].tolist()
        
        # Convert pixel strings to arrays
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(48, 48)
            # Resize to 224x224 for ResNet50
            face = cv2.resize(face, (self.img_size, self.img_size))
            # Convert to 3 channels
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            faces.append(face)
        
        faces = np.array(faces)
        emotions = np.array(emotions)
        
        # Normalize pixel values
        faces = faces / 255.0
        
        print(f"Dataset loaded: {faces.shape[0]} images")
        return faces, emotions
    
    def prepare_data_splits(self, faces, emotions, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            faces, emotions, test_size=test_size, random_state=42, stratify=emotions
        )
        
        # Second split: separate validation set from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self):
        """Build transfer learning model using ResNet50"""
        print("Building model with ResNet50 backbone...")
        
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build custom top layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully!")
        return model
    
    def create_data_generators(self):
        """Create data augmentation generators"""
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()  # No augmentation for validation
        
        return train_datagen, val_datagen
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        print("Starting training...")
        
        # Create data generators
        train_datagen, val_datagen = self.create_data_generators()
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                'best_emotion_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def fine_tune(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Fine-tune the model by unfreezing some layers"""
        print("Fine-tuning model...")
        
        # Unfreeze the last 20 layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create data generators
        train_datagen, val_datagen = self.create_data_generators()
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                'best_emotion_model_finetuned.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Continue training
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print("Evaluating model...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved as 'training_history.png'")
    
    def save_model(self, filepath='emotion_detection_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")


def main():
    """Main training pipeline"""
    # Initialize model
    emotion_model = EmotionDetectionModel(dataset_path='fer2013.csv')
    
    # Load and preprocess data
    faces, emotions = emotion_model.load_and_preprocess_data()
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = emotion_model.prepare_data_splits(
        faces, emotions
    )
    
    # Build model
    emotion_model.build_model()
    
    # Print model summary
    print("\nModel Summary:")
    emotion_model.model.summary()
    
    # Train model
    history = emotion_model.train(
        X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Fine-tune model
    history_ft = emotion_model.fine_tune(
        X_train, y_train, X_val, y_val,
        epochs=20,
        batch_size=32
    )
    
    # Evaluate model
    emotion_model.evaluate(X_test, y_test)
    
    # Plot training history
    emotion_model.plot_training_history(history)
    
    # Save final model
    emotion_model.save_model('emotion_detection_model_final.h5')
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
