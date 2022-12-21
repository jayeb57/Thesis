"""
The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)

"""

import pandas as pd
import os
import shutil

skin_df = pd.read_csv('H:/DATASETS/new10k/HAM10000_metadata.csv')

# Dump all images into a folder and specify the path:
data_dir ="H:/DATASETS/new10k/masks/"

# Path to destination directory where we want subfolders
dest_dir ="H:/DATASETS/new10k/reorganized/"

# Read the csv file containing image names and corresponding labels
skin_df2 = pd.read_csv('H:/DATASETS/new10k/HAM10000_metadata.csv')
print(skin_df['dx'].value_counts())

label=skin_df2['dx'].unique().tolist()  #Extract labels into a list
label_images = []


# Copy images to new folders
for i in label:
    os.mkdir(dest_dir + str(i) + "/")
    sample = skin_df2[skin_df2['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".png"), (dest_dir + i + "/"+id+".png"))
    label_images=[]    

#Now we are ready to work with images in subfolders
    
### FOR Keras datagen ##################################
#flow_from_directory Method
#useful when the images are sorted and placed in there respective class/label folders
#identifies classes automatically from the folder name. 
# create a data generator
    
