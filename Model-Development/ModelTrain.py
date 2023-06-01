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


# Setting Seeds for different libraries



class TrainModel:
    def __init__(self, architecture:str, batch_size:int, image_size:int, validation_split:float, learning_rate:float, seed_n:int, verbose:int, home_dir="/home/ceg98/Documents/"):
        np.random.seed(seed_n)
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

    def format_data(self):

        # TODO: This function needs serious cleaning.
        df = pd.read_csv(self.home_dir + "archive/artists.csv")
        artists = df.sort_values(by=['paintings'], ascending=False)

        # Sort artists by number of paintings
        artists = df.sort_values(by=['paintings'], ascending=False)
        weighted_artists = artists[['name', 'paintings']]
        weighted_artists['weights'] = weighted_artists.paintings.sum() / (weighted_artists.shape[0] * weighted_artists.paintings)

        weighted_artists.head()

        # # Create a dataframe with artists having more than 200 paintings
        artists_top = df[df['paintings'] >= 200].reset_index()
        artists_top = artists_top[['name', 'paintings']]
        artists_top['class_weight'] = max(artists_top.paintings)/artists_top.paintings
        artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
        artists_top['weights'] = artists_top['class_weight']
        self.weighted_artists = artists_top

        self.class_weights = self.weighted_artists['weights'].to_dict() # TODO: Check if this causing a bug. 

        # Explore images of top artists
        images_dir = self.home_dir + "archive/resized/resized"
        self.artists_dirs = os.listdir(images_dir)
        self.artists_name = self.weighted_artists['name'].str.replace(' ', '_').values



        # Specify the directory path containing the image files
        self.images_dir = self.home_dir + "archive/resized/resized"

        # Get a list of all image files in the directory
        self.image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]


        # for artist in self.artists_name:
        #     artist_folder = os.path.join(images_dir, artist)

        #     if not os.path.exists(artist_folder):
        #         os.makedirs(artist_folder)
            
        #     for image_file in image_files: 
        #         source_path = os.path.join(images_dir, image_file)
        #         if artist in source_path:
        #             destination_path = os.path.join(artist_folder, image_file)
        #             shutil.move(source_path, destination_path)

        # See if all directories exist
        self.n_classes = self.weighted_artists.shape[0]

    def create_generators(self):
        self.train_datagen = ImageDataGenerator(validation_split=0.2,
                                        rescale=1./255.,
                                        #   rotation_range=45,
                                        #    width_shift_range=0.1,
                                        #    height_shift_range=0.1,
                                        #   shear_range=5,
                                        #  zoom_range=0.7,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        )

        print("Artists", len(self.artists_name))
        self.train_generator = self.train_datagen.flow_from_directory(directory=self.images_dir,
                                                            class_mode='categorical',
                                                            target_size=self.train_input_shape[0:2],
                                                            batch_size=self.batch_size,
                                                            subset="training",
                                                            shuffle=True,
                                                            classes=self.artists_name.tolist()
                                                        )

        self.valid_generator = self.train_datagen.flow_from_directory(directory=self.images_dir,
                                                            class_mode='categorical',
                                                            target_size=self.train_input_shape[0:2],
                                                            batch_size=self.batch_size,
                                                            subset="validation",
                                                            shuffle=True,
                                                            classes=self.artists_name.tolist()
                                                        )
        
        self.STEP_SIZE_TRAIN = self.train_generator.n//self.train_generator.batch_size
        self.STEP_SIZE_VALID = self.valid_generator.n//self.valid_generator.batch_size

    def get_architecture(self):
        if self.archictecture == "ResNet50":
            self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.train_input_shape)
            self.transfer_learning = True
        # elif self.archictecture == "VGG16":
        #     self.base_model = VGG16

    def define_architecture(self):
        # Load pre-trained model
        
        if self.transfer_learning: 
            for layer in self.base_model.layers:
                layer.trainable = True

            # Add layers at the end
            X = self.base_model.output
            X = Flatten()(X)

            X = Dense(512, kernel_initializer='he_uniform')(X)
            X = Dropout(0.5)(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)

            X = Dense(16, kernel_initializer='he_uniform')(X)
            X = Dropout(0.5)(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)

            output = Dense(self.n_classes, activation='softmax')(X)

            self.model = Model(inputs=self.base_model.input, outputs=output)

    def short_model(self):
        n_epoch = 10
        self.early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                                    verbose=1, mode='auto')

        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, 
                    metrics=['accuracy'])

        # Train the model - all layers
        self.class_weights = self.weighted_artists['weights'].to_dict()

        self.short_history = self.model.fit_generator(generator=self.train_generator, 
                                    steps_per_epoch=self.STEP_SIZE_TRAIN,
                                    validation_data=self.valid_generator, 
                                    validation_steps=self.STEP_SIZE_VALID,
                                    epochs=n_epoch,
                                    shuffle=True,
                                    verbose=self.verbose,
                                    callbacks=[self.reduce_lr],
                                    use_multiprocessing=True,
                                    workers=16,
                                    class_weight=self.class_weights
                                    )
    
    def full_model(self):
        # Freeze core ResNet layers and train again 
        for layer in self.model.layers:
            layer.trainable = False

        for layer in self.model.layers[:50]:
            layer.trainable = True

        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, 
                    metrics=['accuracy'])

        n_epoch = 50
        self.full_history = self.model.fit_generator(generator=self.train_generator, steps_per_epoch=self.STEP_SIZE_TRAIN,
                                    validation_data=self.valid_generator, validation_steps=self.STEP_SIZE_VALID,
                                    epochs=n_epoch,
                                    shuffle=True,
                                    verbose=self.verbose,
                                    callbacks=[self.reduce_lr, self.early_stop],
                                    use_multiprocessing=True,
                                    workers=16,
                                    class_weight=self.class_weights
                                    )
        
        


    def train(self):
        self.format_data()
        self.create_generators()
        self.get_architecture()
        self.define_architecture()
        self.short_model()
        
        history = {}
        history['loss'] = self.short_history.history['loss'] 
        history['accuracy'] = self.short_history.history['accuracy'] 
        history['val_loss'] = self.short_history.history['val_loss'] 
        history['val_accuracy'] = self.short_history.history['val_accuracy'] 
        history['lr'] = self.short_history.history['lr']

        
        history['arch'] = self.archictecture
        history['batch-size'] = self.batch_size
        history['image-size'] = self.image_size
        history['learning-rate'] = self.lr
        history['seed'] = self.seed_n
        history['valid_split'] = self.validation_split

        if self.short_history.history['accuracy'][-1] > 0.14: # Making sure first couple layers is atleast above 50% accuracy.
            self.full_model()

            history[''] += self.full_history.history['loss']
            history[' '] += self.full_history.history['accuracy']
            history[' '] += self.full_history.history['val_loss']
            history[' '] += self.full_history.history['val_accuracy']
            history[' '] += self.full_history.history['lr']
        
        return history
