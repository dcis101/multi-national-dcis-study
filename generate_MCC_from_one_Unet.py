## Created Date: October 20, 2025
## Name: Rui Hou
## Study: Cross-national radiomics validation study using mammography to predict occult invasion in ductal carcinoma in situ
## Requirement: Python 2.7
## The model was originally trained with python 2.7 and keras with tensorflow backend. 

import pydicom,glob,png,cv2,os,sys
import numpy as np
from imageio import imsave
from imageio.v2 import imread
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
from keras.models import *
from keras.layers import *
from PIL import Image
import matplotlib.pyplot as plt
import skimage
n_channel = 3

def write_as_png(one_save_path,one_patch):
	with open(one_save_path, 'wb') as f:
		writer = png.Writer(width=one_patch.shape[1], height=one_patch.shape[0], bitdepth=16, greyscale=True)
		zgray = one_patch
		zgray2list = zgray.tolist()
		writer.write(f, zgray2list)


if __name__ == '__main__':
	one_dcm_full_path = sys.argv[1]  # Original DICOM File Path
	one_lesion_full_path = sys.argv[2]# Radiologist masked lesion ROI
	one_breast_region_full_path = sys.argv[3]# Magnification views breast only region ROI (no paddles), FFDM's breast region
	# print('Start Case (%d):%s ' %(ids, one_dcm_full_path.split('/')[-1]))

	ds = pydicom.read_file(one_dcm_full_path)
	roi_img = cv2.dilate(imread(one_lesion_full_path),np.ones((12,12)))
	breast_img = imread(one_breast_region_full_path)
	breast_img = cv2.erode(breast_img, np.ones((50, 50)))
	roix = np.where(roi_img>0)[0];roiy = np.where(roi_img>0)[1]
	dcm = ds.pixel_array
	##
	breast_region = dcm[np.where(breast_img>0)]
	breast_min = int(breast_region.min())
	breast_max = int(breast_region.max())
	# compress_ratio_latent64 = 512/8
	compress_ratio_latent64 = 8
	dcm_roi_enlarge_shape0 = ((roix.max()-roix.min())//compress_ratio_latent64+1)*compress_ratio_latent64
	dcm_roi_enlarge_shape1 = ((roiy.max()-roiy.min())//compress_ratio_latent64+1)*compress_ratio_latent64
	enlarged_shape = dcm_roi_enlarge_shape0 if dcm_roi_enlarge_shape0 > dcm_roi_enlarge_shape1 else dcm_roi_enlarge_shape1
	##
	limx = [max(0, int(roix.mean())-enlarged_shape//2), min(int(roix.mean())+enlarged_shape//2, dcm.shape[0])]
	limy = [max(0, int(roiy.mean())-enlarged_shape//2), min(int(roiy.mean())+enlarged_shape//2, dcm.shape[1])]
	dcm_roi_enlarge = dcm[limx[0]:limx[1],limy[0]:limy[1]]
	dcm_roi_enlarge = dcm_roi_enlarge.astype('float32')
	dcm_roi_enlarge = (dcm_roi_enlarge-breast_min)/(breast_max-breast_min)
	##
	dcm_input = np.zeros([enlarged_shape, enlarged_shape]).astype('float32')
	dcm_input = dcm_input.reshape((1, dcm_input.shape[0], dcm_input.shape[1],1))

	dcm_input[0,:dcm_roi_enlarge.shape[0], :dcm_roi_enlarge.shape[1],0] = np.copy(dcm_roi_enlarge)  # 人工cut
	######################################################################################################
	dcm_input = np.tile(dcm_input, (1, 1, 1, n_channel))
	inputs = Input(shape = (dcm_input.shape[1],dcm_input.shape[2],n_channel))
	##
	conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	up5 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
	merge5 = concatenate([conv3,up5], axis = 3)
	conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
	conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
	merge6 = concatenate([conv2,up6], axis = 3)
	conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
	up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv1,up7], axis = 3)
	conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
	conv7 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
	unet_output1 = Conv2D(1, 1, activation='sigmoid')(conv7)
	model_unet1 = Model(inputs=inputs, outputs=unet_output1)
	######################################################################################################
	new_mcc_mask1 = np.zeros(dcm.shape)
	selected_weights1 = './weights_initialLR0.001+.260-0.1997.hdf5'
	model_unet1.load_weights(selected_weights1)  # save
	dcm_predict2 = model_unet1.predict(dcm_input)
	dcm_predict_crop2 = dcm_predict2[0, :dcm_roi_enlarge.shape[0], :dcm_roi_enlarge.shape[1], 0]
	if np.unique(dcm_predict_crop2).shape[0] > 1:
		print(skimage.filters.threshold_otsu(dcm_predict_crop2))
		new_mcc_mask1[limx[0]:limx[1], limy[0]:limy[1]] = 255*((dcm_predict_crop2 >= skimage.filters.threshold_otsu(dcm_predict_crop2)).astype(np.uint8))
		new_mcc_mask1[np.where(roi_img==0)] = 0
		print('---Done Weights2 Extracting %s ' % one_dcm_full_path.split('/')[-1])
	else:
		print('---Case %s fail For Segmentation MCC, check your image' % one_dcm_full_path.split('/')[-1])

	######################################################################################################
	img_name = one_dcm_full_path.split('/')[-1].split('.dcm')[0]
	imsave(img_name+'_mcc.png', new_mcc_mask1)
	