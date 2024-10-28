import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pydicom,dicom,glob,csv,glob,random,time,cv2,sys,os,itertools,math,openpyxl,copy,shutil # xlrd
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters,measure
# from skimage.measure import regionprops,label
from sklearn.metrics import mean_squared_error, roc_curve, auc,roc_auc_score
from skimage.morphology import convex_hull_image
from skimage import filters,io, exposure, img_as_uint, img_as_float
from skimage.filters import threshold_otsu
from scipy.spatial.distance import pdist,cdist
from skimage.feature.texture import greycomatrix,greycoprops
from imageio import imsave,imread
##
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_cluster_features(cluster_Mask,ind_Mask,readImg,res):
	pixel_distance = math.floor(thres_distance/res);
	regionMask_clu = measure.regionprops(cluster_Mask)
	cluster_Area = float(regionMask_clu[0].area)*res*res
	cluster_Ecct = regionMask_clu[0].eccentricity
	cluster_size = float(regionMask_clu[0].major_axis_length)*res
	lab_mask = measure.label(ind_Mask>0)
	regionMask_ind= measure.regionprops(lab_mask)
	Points=[regionMask_ind[i].centroid for i in range(np.unique(lab_mask).shape[0]-1)]
	orig_Graph = cdist(Points,Points)
	no_Edges = float((np.where(orig_Graph.ravel()>0)[0].shape[0])/2) # equal to all distances of one mc to the rest calcs without duplicates = (n-1)+(n-2)+...+1
	cluster_MCsNo = float(len(regionMask_ind))
	cluster_MCsCover = float(np.where(ind_Mask.ravel()>0)[0].shape[0])/float(np.where(cluster_Mask.ravel()>0)[0].shape[0])
	# cluster_Dens = 2.0*no_Edges/(float(cluster_MCsNo)*float(cluster_MCsNo-1))
	rev_ind_Mask = (1-ind_Mask)*cluster_Mask
	cluster_FG_Pix_mean = np.mean(readImg[np.where(ind_Mask>0)])
	cluster_FG_Pix_std = np.std(readImg[np.where(ind_Mask>0)])
	# cluster_BG_Pix_mean = np.mean(readImg[np.where(rev_ind_Mask>0)])
	# cluster_BG_Pix_std = np.std(readImg[np.where(rev_ind_Mask>0)])
	bbox_cluster = [int(regionMask_clu[0].bbox[i]) for i in range(4)]
	## Changed Two bg features
	dilate_pix = int(min(bbox_cluster[2]-bbox_cluster[0],bbox_cluster[3]-bbox_cluster[1]))
	cluster_Mask_dilated = cv2.dilate(cluster_Mask,np.ones((dilate_pix,dilate_pix)))
	surrounding_Mask = cluster_Mask_dilated - cluster_Mask
	cluster_BG_Pix_mean_contrast = float(np.mean(readImg[np.where(rev_ind_Mask>0)]))/float(np.mean(readImg[np.where(surrounding_Mask>0)]))
	cluster_BG_Pix_std_contrast = float(np.std(readImg[np.where(rev_ind_Mask>0)]))/float(np.std(readImg[np.where(surrounding_Mask>0)]))
	##
	center_Patch=readImg[bbox_cluster[0]:bbox_cluster[2],bbox_cluster[1]:bbox_cluster[3]]
	## GLCM check papers with glcm on mammography
	g = greycomatrix(center_Patch, distances = [1], angles =[0, np.pi/4,np.pi/2,3*np.pi/4], levels=4096, normed=True, symmetric=True)
	ContrastMask = greycoprops(g,'contrast')
	meanContrastMask = ContrastMask.mean()
	CorrelationMask = greycoprops(g,'correlation')
	meanCorrelationMask = CorrelationMask.mean()
	EnergyMask = greycoprops(g,'energy')
	meanEnergyMask = EnergyMask.mean()
	HomoMask = greycoprops(g,'homogeneity')
	meanHomoMask = HomoMask.mean()
	cluster_features = np.array([cluster_Area, cluster_Ecct, cluster_MCsNo, cluster_MCsCover, cluster_BG_Pix_mean_contrast, cluster_BG_Pix_std_contrast, cluster_FG_Pix_mean, cluster_FG_Pix_std, meanContrastMask, meanCorrelationMask, meanEnergyMask, meanHomoMask,cluster_size])
	return cluster_features


def extract_individual_features(cluster_Mask,ind_Mask,readImg,res,window_size):
	# ind_pix_min = 5
	regionMask_clu = measure.regionprops(cluster_Mask)
	lab_mask_init = measure.label(ind_Mask)
	regionMask_ind= measure.regionprops(lab_mask_init)
	### ADDED to AVOID NAN
	for idss,one_ind in enumerate(np.unique(lab_mask_init)[1:]):
		label_now = regionMask_ind[idss].label
		yStart=max(0,int(round(regionMask_ind[idss].centroid[0]-window_size/2)))
		yEnd=min(lab_mask_init.shape[0],int(math.floor(regionMask_ind[idss].centroid[0]+window_size/2)))
		xStart=max(0,np.int(round(regionMask_ind[idss].centroid[1]-window_size/2)))
		xEnd=min(lab_mask_init.shape[1],int(math.floor(regionMask_ind[idss].centroid[1]+window_size/2)))
		one_ind_mask = (lab_mask_init==label_now).astype(np.uint8)
		center_mask_ind=one_ind_mask[yStart:yEnd,xStart:xEnd]
		if np.unique(center_mask_ind).shape[0]==1:
			lab_mask_init[np.where(lab_mask_init==label_now)]=0
	##		
	lab_mask = measure.label(lab_mask_init>0)
	regionMask_ind= measure.regionprops(lab_mask)
	pixel_distance = math.floor(thres_distance/res);
	pixel_distance_for_normalized_degree = regionMask_clu[0].major_axis_length # Must be values >= max MCs distances
	###
	inds_Peri = np.array([regionMask_ind[i].perimeter*res for i in range(np.unique(lab_mask).shape[0]-1)]).reshape(-1,1)
	inds_Area = np.array([regionMask_ind[i].area*res*res for i in range(np.unique(lab_mask).shape[0]-1)]).reshape(-1,1)
	inds_Cir = np.array([4.0*math.pi*regionMask_ind[i].area/np.square(regionMask_ind[i].perimeter) for i in range(np.unique(lab_mask).shape[0]-1)]).reshape(-1,1)
	inds_Ecc = np.array([regionMask_ind[i].eccentricity for i in range(np.unique(lab_mask).shape[0]-1)]).reshape(-1,1)
	inds_Maxaxis = np.array([regionMask_ind[i].major_axis_length*res for i in range(np.unique(lab_mask).shape[0]-1)]).reshape(-1,1)
	inds_Minaxis = np.array([regionMask_ind[i].minor_axis_length*res for i in range(np.unique(lab_mask).shape[0]-1)]).reshape(-1,1)
	##
	Points=[regionMask_ind[i].centroid for i in range(np.unique(lab_mask).shape[0]-1)]
	orig_Graph = cdist(Points,Points)
	# orig_Graph[np.where(orig_Graph>pixel_distance)]=0
	no_Edges = float((np.where(orig_Graph.ravel()>0)[0].shape[0])/2)
	P2C_Graph = cdist(Points,np.array(regionMask_clu[0].centroid).reshape(1,2)) # Each individual MC to cluster centroid distances
	ind_features_now = np.concatenate((inds_Peri, inds_Area, inds_Cir, inds_Ecc, inds_Maxaxis, inds_Minaxis),axis = 1)
	ind_features_loop = np.zeros([np.unique(lab_mask)[1:].shape[0],18])
	for idss,one_ind in enumerate(np.unique(lab_mask)[1:]):
		label_now = regionMask_ind[idss].label
		inds_D2C = P2C_Graph[idss,0] # Distance to cluster centroid
		inds_D2D_all = orig_Graph[idss,:] # Distances to all the other individual MCs
		inds_D2D = np.sort(inds_D2D_all[np.where(inds_D2D_all>0)])[0] # Distance to nearest MC
		##
		inds_degree = inds_D2D_all[np.where(inds_D2D_all>0)].shape[0] #inds_D2D_all.sum() 
		inds_Ndegree = (1-(inds_D2D_all[np.where(inds_D2D_all>0)]/pixel_distance_for_normalized_degree)).sum()
		yStart=max(0,int(round(regionMask_ind[idss].centroid[0]-window_size/2)))
		yEnd=min(lab_mask.shape[0],int(math.floor(regionMask_ind[idss].centroid[0]+window_size/2)))
		xStart=max(0,np.int(round(regionMask_ind[idss].centroid[1]-window_size/2)))
		xEnd=min(lab_mask.shape[1],int(math.floor(regionMask_ind[idss].centroid[1]+window_size/2)))
		##
		one_ind_mask = (lab_mask==label_now).astype(np.uint8)
		center_mask_ind=one_ind_mask[yStart:yEnd,xStart:xEnd]
		center_mask_background=1-one_ind_mask[yStart:yEnd,xStart:xEnd]
		center_patch_ind = readImg[yStart:yEnd,xStart:xEnd]
		moments = cv2.moments(center_mask_ind)
		huMoments = cv2.HuMoments(moments).reshape(7,)
		##
		inds_FG_Pix_mean = center_patch_ind[np.where(center_mask_ind>0)].mean()
		inds_FG_Pix_std = center_patch_ind[np.where(center_mask_ind>0)].std()
		inds_BG_Pix_mean = center_patch_ind[np.where(center_mask_background>0)].mean()
		inds_BG_Pix_std = center_patch_ind[np.where(center_mask_background>0)].std()
		##
		glcm_ind = greycomatrix(center_patch_ind, distances = [1], angles =[0, np.pi/4,np.pi/2,3*np.pi/4], levels=4096, normed=True, symmetric=True)
		meanContrastMask_ind = greycoprops(glcm_ind,'contrast').mean()
		meanCorrelationMask_ind = greycoprops(glcm_ind,'correlation').mean()
		meanEnergyMask_ind = greycoprops(glcm_ind,'energy').mean()
		meanHomoMask_ind = greycoprops(glcm_ind,'homogeneity').mean()
		##
		ind_feat_topology = np.array([inds_D2C,inds_D2D,inds_Ndegree])
		ind_feat_texture = np.array([inds_FG_Pix_mean,inds_FG_Pix_std,inds_BG_Pix_mean,inds_BG_Pix_std,meanContrastMask_ind, meanCorrelationMask_ind, meanEnergyMask_ind, meanHomoMask_ind])
		ind_features_loop[idss,:] = np.concatenate((huMoments,ind_feat_topology,ind_feat_texture))
	##	
	individual_features = np.concatenate((ind_features_now,ind_features_loop),axis = 1)
	nan_row,nan_idx = np.where(np.isnan(individual_features)==True)
	if nan_row.shape[0]>0:
		individual_features = np.delete(individual_features,nan_row,0)
		print('CAUTIONS: Nan Features IN Individual Features')
	return individual_features

if __name__ == '__main__':
	thres_distance = 10.0
	thres_window = 1.0
	ind_area_min = 0.05#0.1 # Perhaps change to about 12pix
	ind_area_max = 4
	ind_cir_thres = [0.9,1.1] # and area>1
	ind_major_to_minor_thres = 5
	cluster_features_names = ['MCC Area', 'MCC Eccentricity', 'MCC NO of MCs',  'MCC MCs Coverage', 'MCC Background Mean', 'MCC Background Std', 'MCC All MCs Intensity Mean', 'MCC All MCs Intensity Std', 'Mean GLCM Contrast', 'Mean GLCM Correlation', 'Mean GLCM Energy', 'Mean GLCM Homogeneity', 'Lesion Size']
	ind_features_names = ['MC Perimeter', 'MC Area', 'MC Circularity', 'MC Eccentricity', 'MC Major Axis', 'MC Minor Axis', '1st Variant of Hu Moments','2nd Variant of Hu Moments','3rd Variant of Hu Moments','4th Variant of Hu Moments','5th Variant of Hu Moments','6th Variant of Hu Moments','7th Variant of Hu Moments','MC Distance to Cluster Centroid','MC Distance to Nearest MC','MC Normalized Degree','MC Intensity Mean','MC Intensity Std','MC Background Intensity Mean','MC Background Intensity Std','MC Mean GLCM Contrast','MC Mean GLCM Correlation','MC Mean GLCM Energy','MC Mean GLCM Homogeneity']
	###
	case_root_dir = ''## Path-to-dicom
	dcms_all_one_sub_dir = glob.glob(os.path.join(case_root_dir,'DICOM','*.dcm'))
	for ids, one_dcm_path in enumerate(dcms_all_one_sub_dir):
		one_roi_path = one_dcm_path.replace('/DICOM/','/ROI/').replace('.dcm','.png')
		one_feat_path = one_dcm_path.replace('/DICOM/','/Radiomics_feature109/').replace('.dcm','_feat.npy')
		##
		one_mcc_path = one_dcm_path.replace('/DICOM/','/MCC/').replace('.dcm','_MCC.png')
		try:
			ds = dicom.read_file(one_dcm_path)
		except:
			ds = pydicom.read_file(one_dcm_path)
		readImg = ds.pixel_array
		# if 'ImagerPixelSpacing' in ds.dir():
		res = float(ds.ImagerPixelSpacing[0])
		ind_pix_min = int(math.floor(ind_area_min/res/res))
		ind_pix_max = int(math.floor(ind_area_max/res/res))
		pixel_distance = math.floor(thres_distance/res)
		window_size = math.floor(thres_window/res)
		window_size = round((window_size-1)/2)*2+1;
		try: 
			ind_Mask_init  = imread(one_mcc_path)
		except:
			print('--###--###--##--MCC with UNET not available, skip %s (index %d)'%(one_dcm_path.split('/')[-1], ids))
			continue
		# ind_Mask /=255
		lab_mask_init = measure.label(ind_Mask_init)
		lab_ind,counts_ind = np.unique(lab_mask_init,return_counts  = True)
		lab_too_small = lab_ind[np.where(counts_ind<ind_pix_min)]
		lab_too_big = lab_ind[np.where(counts_ind>ind_pix_max)]
		for one_remove in np.concatenate((lab_too_big,lab_too_small)):
			lab_mask_init[np.where(lab_mask_init==one_remove)]=0
		##
		ind_Mask = (lab_mask_init>0).astype(np.uint8)
		cluster_Mask = convex_hull_image(ind_Mask).astype(np.uint8)
		########
		try:
			cluster_features = extract_cluster_features(cluster_Mask,ind_Mask,readImg,res)
			individual_features_all = extract_individual_features(cluster_Mask,ind_Mask,readImg,res,window_size)
		except:
			print('________UNET-MCC FAIL FEATURE Extraction %s.'%(one_dcm_path.split('/')[-1]))
			one_mcc_path = one_dcm_path.replace('/DICOM/','/MCC_Masks_with_CV/').replace('.dcm','_MCC.png')
			try: 
				ind_Mask_init  = imread(one_mcc_path)
			except:
				print('--###--###--##--MCC with CV not available, skip %s (index %d)'%(one_dcm_path.split('/')[-1], ids))
				continue
			# ind_Mask /=255
			lab_mask_init = measure.label(ind_Mask_init)
			lab_ind,counts_ind = np.unique(lab_mask_init,return_counts  = True)
			lab_too_small = lab_ind[np.where(counts_ind<ind_pix_min)]
			lab_too_big = lab_ind[np.where(counts_ind>ind_pix_max)]
			for one_remove in np.concatenate((lab_too_big,lab_too_small)):
				lab_mask_init[np.where(lab_mask_init==one_remove)]=0		
			ind_Mask = (lab_mask_init>0).astype(np.uint8)
			cluster_Mask = convex_hull_image(ind_Mask).astype(np.uint8)
			try:
				cluster_features = extract_cluster_features(cluster_Mask,ind_Mask,readImg,res)
				individual_features_all = extract_individual_features(cluster_Mask,ind_Mask,readImg,res,window_size)
			except:
				print('________CV-MCC ALSO FAIL FEATURE Extraction %s: '%(one_dcm_path.split('/')[-1]))
				continue
		# Lesion Size was the last feature
		feature_Matrix = np.concatenate((cluster_features[:-1],individual_features_all.mean(0),individual_features_all.std(0),individual_features_all.min(0),individual_features_all.max(0),cluster_features[-1:]))
		np.save(one_feat_path,feature_Matrix)
		print('***********Finished %s (index %d): '%(one_dcm_path.split('/')[-1], ids))
	########################################################################################################################################################################
	########################################################################################################################################################################
