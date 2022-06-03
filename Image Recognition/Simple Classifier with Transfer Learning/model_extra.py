''' Inceasing the accuracy of the classifier '''

# The ResNet50 model's output is going to be connected to this classifier.

av1 = GlobalAveragePooling2D()(model.output)
fc1 = Dense(256, activation = 'relu')(av1)
d1 = Dropout(0.5)(fc1)
fc2 = Dense(128, activation = 'relu')(d1)
d2 = Dropout(0.5)(fc2)
fc3 = Dense(64, activation = 'relu')(d2)
d3 = Dropout(0.5)(fc3)
fc4 = Dense(10, activation = 'softmax')(d3)

''' MobileNet over ResNet '''

# Import the required libraries
from tensorflow.python.keras.applications import MobileNet

# Load the MobileNet model
mobile_net_model = MobileNet(include_top=False, 
                             weights='imagenet',
                             input_shape = (224,224,3))
# mobile_net_model.summary()

# Create the classifier and connect it to the MobileNet model
av1 = GlobalAveragePooling2D()(mobile_net_model.output)
fc1 = Dense(256, activation = 'relu')(av1)
d1 = Dropout(0.5)(fc1)
fc2 = Dense(128, activation = 'relu')(d1)
d2 = Dropout(0.5)(fc2)
fc3 = Dense(64, activation = 'relu')(d2)
d3 = Dropout(0.5)(fc3)
fc4 = Dense(10, activation = 'softmax')(d3)
mobile_net_final_model = Model(inputs = mobile_net_model.input, 
                               outputs = fc4)
# mobile_net_final_model.summary()

# Compile the model
adam = Adam(lr = 0.00003)
mobile_net_final_model.compile(loss = 'categorical_crossentropy', 
                               optimizer = adam, 
                               metrics = ['accuracy'])

# Freeze the training process for the first 50 layers of MobileNet model
for ix in range(50):
    mobile_net_final_model.layers[ix].trainable = False
print(mobile_net_final_model.summary())

# Train the model
hist = mobile_net_final_model.fit(X_train, Y_train, 
                                  shuffle = True, 
                                  batch_size = 16, 
                                  epochs = 8, 
                                  validation_split = 0.20)

from tensorflow.python.keras.applications.mobilenet import preprocess_input

img_path = 'image.png'
def load_img(path):
    img = image.load_img(path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    return x
  
x = load_img(img_path)
pred = mobile_net_final_model.predict(x)

# get integer corresponding to particular class
print(np.argmax(pred)) 
