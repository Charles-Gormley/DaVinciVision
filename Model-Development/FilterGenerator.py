from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import os
import imgaug.augmenters as iaa
from tensorflow.keras.utils import to_categorical

class CustomGenerator(Sequence):
    def __init__(self, directory, batch_size, classes:list, laplacian=False, garbor=False, augments=False):
        self.directory = directory
        self.batch_size = batch_size
        self.laplacian = laplacian
        self.garbor = garbor
        self.augments = augments
        self.classes = classes
        self.augmenter = iaa.Sequential([
            iaa.Affine(rotate=(-20, 20)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255))
        ])
        self.class_names = self.classes
        self.class_mapping = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.file_list = []
        self.labels = []
        for class_name in self.class_names:
            class_path = os.path.join(directory, class_name)
            files = os.listdir(class_path)
            self.file_list.extend([os.path.join(class_path, file) for file in files])
            self.labels.extend([self.class_mapping[class_name]] * len(files))
        self.labels = to_categorical(self.labels, num_classes=(len(self.class_names)))

    def load_and_preprocess_image(self, image_path):
        # Strip image path
        image = cv2.imread(image_path)
        
        

        # Preprocess the image as needed
        # Example: Convert color channels (e.g., BGR to RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (225, 225))
        preproccesed_image = np.array(image_rgb)


        # Image pathing
        sub_image_path = os.path.join(*(image_path.split(os.sep))[-2:])

        if self.laplacian:
            base_lap = "/home/ceg98/Documents/Laplacian/"
            lap_path = base_lap  + sub_image_path
            
            lap_image = cv2.imread(lap_path, cv2.IMREAD_GRAYSCALE)
            lap_image = cv2.resize(lap_image, (225, 225))
            preproccesed_image = np.dstack((preproccesed_image, np.array(lap_image)))
            
        
        if self.garbor:
            base_garbor = "/home/ceg98/Documents/Garbor/"
            garbor_path = base_garbor + sub_image_path
            
            garbor_image = cv2.imread(garbor_path)
            garbor_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            garbor_image = cv2.resize(garbor_image, (225, 225))
            preproccesed_image = np.dstack((preproccesed_image, np.array(garbor_image)))
        
        
        return preproccesed_image
        
        

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size : (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]

        batch_images = []
        for file_name in batch_files:
            image_path = os.path.join(self.directory, file_name)
            # Load the image and perform any necessary preprocessing
            image = self.load_and_preprocess_image(image_path)
            batch_images.append(image)

        
        # Convert the list of images to a numpy array
        batch_images = np.array(batch_images)
        
        # Augment the images
        
        

        if self.augments:
            batch_images = self.augmenter.augment_images(batch_images)
        return batch_images, np.array(batch_labels)

    def __len__(self):
        return len(self.file_list) // self.batch_size