import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Optional: data augmentation
    width_shift_range=0.2,  # Optional: data augmentation
    height_shift_range=0.2,  # Optional: data augmentation
    horizontal_flip=True  # Optional: data augmentation
)

test_datagen = ImageDataGenerator(
    rescale=1./255  # Normalize pixel values
)

train_dir = 'data/training'
test_dir = 'data/testing'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images as needed
    batch_size=20,
    class_mode='sparse'  # Use 'sparse' since you have multiple classes
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),  # Resize images as needed
    batch_size=20,
    class_mode='sparse'
)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))  # Assuming 4 classes

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Adjust according to the number of images
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50  # Adjust according to the number of images
)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(validation_generator, steps=50)
print("Test accuracy:", test_acc)