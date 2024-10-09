

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Now as we have to predict the 10 different diseases from given pictures in cancer_images dataset
#print all the diseases folders and save as list 
diseases = os.listdir('C:\\Users\\lenovo\\Downloads\\cancer images')
print(diseases)
len(diseases)
#Now print all the diseases with labels and its pictures in one image
plt.figure(figsize=(20, 20))
for i, disease in enumerate(diseases):
    for j in range(1):
        img = cv2.imread('C:\\Users\\lenovo\\Downloads\\cancer images/'+disease+'/'+os.listdir('C:\\Users\\lenovo\\Downloads\\cancer images/'+disease)[j])
        plt.subplot(10, 10, i+1)
        plt.imshow(img)
        plt.title(disease)
        plt.xticks([])
        plt.yticks([])
# As we have to predict the diseases using CNN 
# Now we will create the training and testing dataset
# We will create the training and testing dataset
X = []
y = []
for i, disease in enumerate(diseases):
    for img in os.listdir('C:\\Users\\lenovo\\Downloads\\cancer images/'+disease):
        img = cv2.imread('C:\\Users\\lenovo\\Downloads\\cancer images/'+disease+'/'+img)
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(disease)
X = np.array(X)
y = np.array(y)
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Assuming X and y are already defined

# Normalize X
X = X/255

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert labels to categorical
y = to_categorical(y)

#Now we will split the data into training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Initialize the Sequential model
model = Sequential()

# Add Convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Add Fully Connected layers
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Compile the model with hypertuned parameters
optimizer = Adam(learning_rate=0.001)  # Adjust learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
# Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[early_stopping, reduce_lr])
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
#print the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Lets see the other evaluation metrics 

from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))


 # Sauvegarder le modèle
model.save('model_cancer.h5')
