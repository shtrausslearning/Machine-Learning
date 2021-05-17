''' MAIN FUNCTION '''

# Evaluate CNN model w/ imported list of augmentation options
def augment_model(lst_aug):
    
    # Requires list of augmentations to training & val/test data

    # Define DataGenerator, load image data from directory
    gen_train = lst_aug[0].flow_from_directory(train_folder, 
                            target_size=(224,224),  # target size
                            batch_size=32,          # batch size
                            class_mode='categorical')    # batch size

    gen_valid = lst_aug[1].flow_from_directory(val_folder,
                            target_size=(224,224),
                            batch_size=32,
                            class_mode='categorical')

    gen_test = lst_aug[1].flow_from_directory(test_folder,
                            target_size=(224,224),
                            batch_size=32,
                            class_mode='categorical')

    # Define a CNN Model
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", input_shape=sshape),    
        keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(labels, activation="softmax")
    ])
    
    # Compile Model (requires custom functions)
    model.compile(optimizer='Adam',loss='categorical_crossentropy',
                  metrics=['acc',get_f1,get_precision,get_recall])
    
    # Callback Options During Training 
    callbacks = [ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=0, 
                                   factor=0.5,mode='max',min_lr=0.001),
                 ModelCheckpoint(filepath=f'model_out.h5',monitor='val_accuracy',
                                 mode = 'max',verbose=0,save_best_only=True),
                 TqdmCallback(verbose=0)] 
    
    
    # Evaluate Model
    history = model.fit(gen_train,
                        validation_data = gen_valid,
                        callbacks=callbacks,
                        verbose=0,epochs=n_epochs)
    
    # Return Result History
    return history 

''' AUGMENTATION OPTIONS '''

# lst of augmentation options
lst_augopt = ['rescale','horizontal_flip','vertical_flip',
              'brightness_range','rotation_range','shear_range',
              'zoom_range','width_shift_range','height_shift_range',
              'channel_shift_range','zca_whitening','featurewise_center',
              'samplewise_center','featurewise_std_normalization','samplewise_std_normalization']

# lst of default setting
lst_augval = [1.0/255,True,True,  
              [1.1,1.5],0.2,0.2,
              0.2,0,0,
              0,True,False,
             False,False,False]

''' HELPER FUNCTIONS '''

# Get Augmentation Names from lst_select options
def get_aug_name(lst_select):
    lst_selectn = [];
    for i in lst_select:
        tlst_all = []
        for j in i:
            tlist_selectn = tlst_all.append(lst_augopt[j])
        lst_selectn.append(tlst_all)
    return lst_selectn

# Model Evaluation w/ Augmentation
def aug_eval(lst_select=None):

    ii=-1; lst_history = []
    for augs in lst_select:

        print('Augmentation Combination')
        # get dictionary of augmentation options
        ii+=1; dic_select = dict(zip([lst_augopt[i] for i in lst_select[ii]],[lst_augval[i] for i in lst_select[ii]]))
        print(dic_select)

        # define augmentation options
        train_datagen = ImageDataGenerator(**dic_select) # pass arguments
        gen_datagen = ImageDataGenerator(rescale=1.0/255) # evaluation set

        # evaluate model & return history metric
        history = augment_model([train_datagen,gen_datagen])

        # store results
        lst_history.append(history)
        
    return lst_history
  
''' EXAMPLE '''

# Select Augmentation
lst_select = [[0,1],[0,2],[0,3],[0,1,5,6]] # list of augmentations combinations to test
lst_selectn = get_aug_name(lst_select)     # get list of augmentation names
print(lst_selectn)

# Evaluation Augmentation Model
history = aug_eval(lst_select) # get list of results
