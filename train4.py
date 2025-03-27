import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

def create_optimized_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Оптимизированная модель для малого датасета
    """
    # Берем ResNet50V2 (улучшенную версию) без верхних слоев
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'  # Добавляем средний пулинг сразу
    )
    
    # Замораживаем все слои базовой модели
    base_model.trainable = False
    
    # Создаем новую голову модели
    x = base_model.output
    x = Dropout(0.5)(x)  # Добавляем dropout для регуляризации
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(0.01)  # L2 регуляризация
    )(x)
    x = Dropout(0.3)(x)
    predictions = Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=l2(0.01)
    )(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model_with_augmentation(model, train_dir, batch_size=16, epochs=30):
    """
    Улучшенный процесс обучения с аугментацией
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25,  # Больше данных для валидации
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Используем уменьшающийся learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=100,
        decay_rate=0.9
    )
    
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Добавляем раннюю остановку
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // batch_size),
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def main():
    parser = argparse.ArgumentParser(description='Optimized ResNet for small handwriting dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()
    
    # Проверка данных
    if not all(os.path.exists(os.path.join(args.data_dir, gender)) for gender in ['female', 'male']):
        print("Error: Dataset directories not found.")
        return
    
    print("Creating optimized model...")
    model = create_optimized_model()
    
    print("Training with augmentation...")
    train_model_with_augmentation(
        model,
        args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print(f"Saving model to {args.model_path}...")
    model.save(args.model_path)
    print("Training complete!")

if __name__ == '__main__':
    main()