#from simple_unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from keras_unet_collection import models, losses
from tensorflow.keras.optimizers import Adam



#melanoma
image_directory = 'H:/DATASETS/new10k/reorganized_jpg/mel/greyscales/greyscale/'
mask_directory = 'H:/DATASETS/new10k/reorganized_jpg/mel/masks/mel_mask/'


'''image_directory = 'H:/DATASETS/new10k/reorganized_jpg/nv/greyscale/'
mask_directory = 'H:/DATASETS/new10k/reorganized_jpg/nv/mask/'
'''



SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'jpg'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


#Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 1, random_state = 0)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()


IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

print(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
###############################################################


#############################################################


'''Attention-Unet'''

model1 = models.att_unet_2d((256, 256, 1), [64, 128, 256, 512], n_labels=2,
                           stack_num_down=2, stack_num_up=2,
                           activation='ReLU', atten_activation='ReLU', attention='add', output_activation=None, 
                           batch_norm=True, pool=False, unpool='bilinear', name='attunet')

model1.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 1e-3), metrics=['accuracy'])

model1.summary()

history1 = model1.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model1.save('mel_train_unetatt.hdf5')

#############################################################


'''R2U-net'''

model2 = models.r2_unet_2d((256, 256, 1), [16, 32, 64, 128, 256], n_labels=2,
                          stack_num_down=2, stack_num_up=1, recur_num=2,
                          activation='ReLU', output_activation='Softmax', 
                          batch_norm=True, pool='max', unpool='nearest', name='r2unet')

model2.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 1e-3), metrics=['accuracy'])

model2.summary()

history2 = model2.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=5, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model2.save('mel_train_R2Uunet.hdf5')

#############################################################

'''ResUnet-a'''


model3 = models.resunet_a_2d((256, 256, 1), [16, 32, 64, 128, 256], 
                            dilation_num=[1, 3, 15, 31], 
                            n_labels=1, aspp_num_down=128, aspp_num_up=64, 
                            activation='ReLU', output_activation='Sigmoid', 
                            batch_norm=True, pool=False, unpool='nearest', name='resunet')

model3.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 1e-3), metrics=['accuracy'])

model3.summary()

history3 = model3.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=5, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model3.save('mel_train_ResUnet.hdf5')

#############################################################

'''swin_unet'''


model4 = models.swin_unet_2d((256, 256, 1), filter_num_begin=64, n_labels=2, depth=2, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Softmax', shift_window=True, name='swin_unet')


model4.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 1e-3), metrics=['accuracy'])

model4.summary()

history4 = model4.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=5, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model4.save('mel_train_swinUNet.hdf5')

#############################################################

_, acc = model2.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


#plot the training and validation accuracy and loss at each epoch
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#acc = history.history['acc']
acc = history2.history['accuracy']
#val_acc = history.history['val_acc']
val_acc = history2.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##################################
#IOU
y_pred=model2.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#Predict on a few images
model = model2
model.load_weights('mel_train_R2Uunet1.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

test_img_other = cv2.imread('H:/DATASETS/new10k/reorganized_jpg/mel/image/mel_all_res/ISIC_0033559.jpg', 0)
#test_img_other = cv2.imread('H:/DATASETS/new10k/reorganized_jpg/df/image/all/ISIC_0030555.jpg', 0)
#test_img_other = cv2.imread('H:/DATASETS/new10k/reorganized_jpg/nv/image/all/ISIC_0031375.jpg', 0)
#test_img_other = cv2.imread('data/test_images/img8.tif', 0)
test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
test_img_other_input=np.expand_dims(test_img_other_norm, 0)

#Predict and threshold for values above 0.5 probability
#Change the probability threshold to low value (e.g. 0.05) for watershed demo.
prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.49999).astype(np.uint8)

"""prediction_other1 = (model.predict(test_img_other_input)[0,:,:,0] > 0.99).astype(np.uint8)
prediction_other2 = (model.predict(test_img_other_input)[0,:,:,0] > 0.999).astype(np.uint8)"""

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(235)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()