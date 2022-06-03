from tensorflow.keras.preprocessing import image # work with images
from tensorflow.keras.layers import *  # create different layers
from tensorflow.keras.models import *  # create sequential model

# [note]
# Binary Classifier

''' Create Model Architecture '''

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu",input_shape=(224,224,3)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

''' Compile Model '''

model.compile(loss="binary_crossentropy", 
              optimizer="adam",
              metrics = ["accuracy"])
# model.summary()

''' Create Image Data Generators '''
# Add image augmentations during training

train_datagen = image.ImageDataGenerator(rescale=1./255, 
                                         shear_range=0.2, 
                                         zoom_range=0.2, 
                                         horizontal_flip=True)
test_datagen = image.ImageDataGenerator(rescale = 1./255)

# Train/Validation Generators
train_generator = train_datagen.flow_from_directory('CovidDataset/Train',
                                                    target_size=(224,224),
                                                    batch_size=32, 
                                                    class_mode="binary")
val_generator = test_datagen.flow_from_directory('CovidDataset/Val',
                                                 target_size=(224,224),
                                                 batch_size=32, class_mode="binary")

''' Train the Model '''
hist = model.fit(train_generator, 
                 validation_data=val_generator, 
                 epochs = 6, 
                 verbose=0,
                 validation_steps=2)
