# DaVinciVision 🎨
Project for DSCI-471 where the aim is to build CNN prediction models which predict which artist created an artwork.

## Getting Started
1. Pull the Repo. 
2. Download this dataset and **copy the directory where you saved it**.
3. Navigate to the *helpers/SetDataLocation.py* file and change the vlaue for the *path_to_dataset* variables with correct file path splitting for your os ( / - for windows )
4. Run the SetDataLocation.py File. Now you can work with the noteobooks! 👏
5. Check out hte Analysis.ipynb file

## Dataset
https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

| This dataset consists of three files. It is 2.32 GB in size which seems promising to train a quality DL model. There are a total of 8446 images (artworks) and 50 artists (classes). 

* artists.csv: A dataset containing information for each artist.
* images.zip: A collection of full-sized images, divided into folders and sequentially numbered.
* resized.zip: A resized version of the image collection, extracted from the folder structure for faster processing.

## EDA
Looking At HSV Visualization of Van Gogh Painting: 


https://github.com/Charles-Gormley/DaVinciVision/assets/76138796/e9e5f86f-3b6b-4cb3-bdfd-3a7eeb9c0cd3



## Models
### CNN Models 
* AlexNet
* VGG Network(s)
* NiN
* Inception Network(s)
* ResNet & ResNeXt
* DenseNet

## Class Path
- main-directory/
    - class_1/
        - image_1.jpg
        - image_2.jpg
        ...
    - class_2/
        - image_1.jpg
        - image_2.jpg
    etc...
- filtered-directory/
    - class_1-filtered/
        - image_1.jpg
        - image_2.jpg
        ...
    - class_2-filtered/
        - image_1.jpg
        - image_2.jpg
    etc...
etc...