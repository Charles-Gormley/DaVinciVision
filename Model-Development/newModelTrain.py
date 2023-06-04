######### Importing #########
#  Data Science Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Machine Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Internal Libraries
import json
import os
from tqdm import tqdm, tqdm_notebook
import random
import shutil


class TrainModel:
    def __init__(self, architecture:str, batch_size:int, image_size:int, validation_split:float, learning_rate:float, seed_n:int, verbose:int, home_dir="/home/ceg98/Documents/"):
        tf.random.set_seed(seed_n)
        
        self.seed_n = seed_n
        self.archictecture = architecture
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_split = validation_split
        self.lr = learning_rate
        self.images_dir = home_dir + "archive/resized/resized"
        self.train_input_shape = (self.image_size, self.image_size, 3)
        self.home_dir = home_dir
        self.verbose = verbose
        self.n_features = 3
    def train(self):
        df = pd.read_csv("/home/ceg98/Documents/archive/artists.csv")
        artists = df.sort_values(by=['paintings'], ascending=False)
        # Sort artists by number of paintings
        artists = df.sort_values(by=['paintings'], ascending=False)
        weighted_artists = artists[['name', 'paintings']]
        weighted_artists['weights'] = weighted_artists.paintings.sum() / (weighted_artists.shape[0] * weighted_artists.paintings)

        weighted_artists.head()

        # Create a dataframe with artists having more than 200 paintings
        artists_top = df[df['paintings'] >= 200].reset_index()
        artists_top = artists_top[['name', 'paintings']]
        artists_top['class_weight'] = max(artists_top.paintings)/artists_top.paintings
        artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
        artists_top['weights'] = artists_top['class_weight']
        weighted_artists = artists_top
        class_weights = weighted_artists['weights'].to_dict()
        print(class_weights)

        # Explore images of top artists
        images_dir = "/home/ceg98/Documents/archive/resized/resized"
        artists_dirs = os.listdir(images_dir)
        artists_name = weighted_artists['name'].str.replace(' ', '_').values

        # Specify the directory path containing the image files
        images_dir = "/home/ceg98/Documents/archive/resized/resized"

        # Get a list of all image files in the directory
        image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]


        for artist in artists_name:
            artist_folder = os.path.join(images_dir, artist)

            if not os.path.exists(artist_folder):
                os.makedirs(artist_folder)
            
            for image_file in image_files: 
                source_path = os.path.join(images_dir, image_file)
                if artist in source_path:
                    destination_path = os.path.join(artist_folder, image_file)
                    shutil.move(source_path, destination_path)

        for name in artists_name:
            if os.path.exists(os.path.join(images_dir, name)):
                print("Found -->", os.path.join(images_dir, name))
            else:
                print("Did not find -->", os.path.join(images_dir, name))


        # Augment data
        batch_size = 32
        train_input_shape = (225, 225, 3)
        n_classes = weighted_artists.shape[0]

        train_datagen = ImageDataGenerator(validation_split=0.2,
                                        rescale=1./255.,
                                        rotation_range=45,
                                        #    width_shift_range=0.1,
                                        #    height_shift_range=0.1,
                                        #   shear_range=5,
                                        zoom_range=0.7,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        )

        train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                            class_mode='categorical',
                                                            target_size=train_input_shape[0:2],
                                                            batch_size=batch_size,
                                                            subset="training",
                                                            shuffle=True,
                                                            classes=artists_name.tolist()
                                                        )


        valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                            class_mode='categorical',
                                                            target_size=train_input_shape[0:2],
                                                            batch_size=batch_size,
                                                            subset="validation",
                                                            shuffle=True,
                                                            classes=artists_name.tolist()
                                                        )

        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
        print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)

        # Load pre-trained model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

        for layer in base_model.layers:
            layer.trainable = True

        # Add layers at the end
        X = base_model.output
        X = Flatten()(X)

        X = Dense(512, kernel_initializer='he_uniform')(X)
        X = Dropout(0.5)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Dense(16, kernel_initializer='he_uniform')(X)
        X = Dropout(0.5)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        output = Dense(n_classes, activation='softmax')(X)

        model = Model(inputs=base_model.input, outputs=output)

        optimizer = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, 
                    metrics=['accuracy'])
        
        n_epoch = 10

        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                                mode='auto', restore_best_weights=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                                    verbose=1, mode='auto')
        
        # Train the model - all layers
        class_weights = weighted_artists['weights'].to_dict()

        history1 = model.fit_generator(generator=train_generator, 
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    validation_data=valid_generator, 
                                    validation_steps=STEP_SIZE_VALID,
                                    epochs=n_epoch,
                                    shuffle=True,
                                    verbose=1,
                                    callbacks=[reduce_lr],
                                    use_multiprocessing=True,
                                    workers=16,
                                    class_weight=class_weights
                                    )
        for layer in model.layers:
            layer.trainable = False

        for layer in model.layers[:50]:
            layer.trainable = True

        optimizer = Adam(lr=0.0001)

        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, 
                    metrics=['accuracy'])

        n_epoch = 50
        if history1.history['accuracy'][-1] > .4:
            history2 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                        validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                                        epochs=n_epoch,
                                        shuffle=True,
                                        verbose=1,
                                        callbacks=[reduce_lr, early_stop],
                                        use_multiprocessing=True,
                                        workers=16,
                                        class_weight=class_weights
                                        )
            
        history = {}
        history['loss'] = history1.history['loss'] + history2.history['loss']
        history['accuracy'] = history1.history['accuracy'] + history2.history['accuracy']
        history['val_loss'] = history1.history['val_loss'] + history2.history['val_loss']
        history['val_accuracy'] = history1.history['val_accuracy'] + history2.history['val_accuracy']
        history['lr'] = history1.history['lr'] + history2.history['lr']
        return history
