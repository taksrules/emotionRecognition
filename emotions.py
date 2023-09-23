import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Reshape
# Load the emotion dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path='/archive')

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_accuracy)

new_image = cv2.imread("newImage.png")

new_image=  cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

# Resize the new image to be 28x28 pixels in size
resized_image = cv2.resize(new_image, (28, 28))

# Convert the resized image to a NumPy array
resized_image_array = np.array(resized_image)

# Add an extra dimension to the resized image array to represent the channel dimension
resized_image_array = np.expand_dims(resized_image_array, axis=-1)
# resized_image_array = resized_image_array[:, :, :, 0]
# Make a prediction on the resized image
prediction = model.predict(np.expand_dims(resized_image_array, axis=0))
# prediction = model.predict(new_image)
class_index = prediction.argmax()
class_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][class_index]

print('The predicted emotion is:', class_name)
