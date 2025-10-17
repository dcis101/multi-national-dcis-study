## Created Date: October 20, 2025
## Name: Rui Hou
## Study: Cross-national radiomics validation study using mammography to predict occult invasion in ductal carcinoma in situ
## Requirement: Python 2.7
## The model was originally trained with python 2.7 and keras with tensorflow backend. 

import sys, os, random, math, time, glob, csv, copy, openpyxl, datetime, pydicom
import numpy as np
from sklearn import svm, metrics, linear_model, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RandomizedLogisticRegression
from scipy.misc import imread,imsave

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def mean_std_normalize(X_new_tr,X_new_val):
	mean_std_scaler0 = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
	X_tr = mean_std_scaler0.fit_transform(X_new_tr)
	if len(X_new_val)>0:
		X_val = mean_std_scaler0.transform(X_new_val)
	else:
		X_val = []
	return X_tr,X_val

if __name__ == '__main__':
	iterTimes = int(sys.argv[1]) #100
	num_cpus_to_load = 20
	nfold = 5
	embedded_cv = 4
	pow_base_c = 10
	c_min = -10
	c_max = 10
	Cs = [pow(pow_base_c,i) for i in range(c_min,c_max)]
	param_grid = {'C': Cs}
	path_to_save_train_logs = 'training_results/'
	####
	cluster_features_names = ['MCC Area', 'MCC Eccentricity', 'MCC NO of MCs',  'MCC MCs Coverage', 'MCC Background Mean', 'MCC Background Std', 'MCC All MCs Intensity Mean', 'MCC All MCs Intensity Std', 'Mean GLCM Contrast', 'Mean GLCM Correlation', 'Mean GLCM Energy', 'Mean GLCM Homogeneity', 'Lesion Size']
	ind_features_names = ['MC Perimeter', 'MC Area', 'MC Circularity', 'MC Eccentricity', 'MC Major Axis', 'MC Minor Axis', '1st Variant of Hu Moments','2nd Variant of Hu Moments','3rd Variant of Hu Moments','4th Variant of Hu Moments','5th Variant of Hu Moments','6th Variant of Hu Moments','7th Variant of Hu Moments','MC Distance to Cluster Centroid','MC Distance to Nearest MC','MC Normalized Degree','MC Intensity Mean','MC Intensity Std','MC Background Intensity Mean','MC Background Intensity Std','MC Mean GLCM Contrast','MC Mean GLCM Correlation','MC Mean GLCM Energy','MC Mean GLCM Homogeneity']
	radiomics_feat_list =	cluster_features_names[:-1] + [' '.join(['Mean',ss]) for ss in ind_features_names] + [' '.join(['Std',ss]) for ss in ind_features_names] + \
						[' '.join(['Min',ss]) for ss in ind_features_names] + [' '.join(['Max',ss]) for ss in ind_features_names] + \
						cluster_features_names[-1:]

	num_radiomic_features = len(radiomics_feat_list)
	################################## US Cases ##########################
	duke_xlsx_file_name_alldcis = 'DUKE/US-Duke information.xlsx'
	duke_dcis_file = openpyxl.load_workbook(duke_xlsx_file_name_alldcis)
	duke_all= duke_dcis_file.worksheets[0]
	duke_feature_paths_all = glob.glob('DUKE/features_npy/*.npy')
	item_list = [str(duke_all.cell(row = 1, column = col_now).value) for col_now in range(1, duke_all.max_column + 1)]
	dx_col = [col_now + 1 for col_now, ii in enumerate(item_list) if 'Diagnosis' in ii][0]
	model_col = [col_now + 1 for col_now, ii in enumerate (item_list) if 'Manufacturer Model Name' in ii][0]
	fakeid_col = [col_now + 1 for col_now, ii in enumerate(item_list) if 'Fake ID' in ii][0]
	duke_names_with_dx_model_feat_ori = [[str(duke_all.cell(row = idxx,column = fakeid_col).value),str(duke_all.cell(row = idxx,column = dx_col).value),str(duke_all.cell(row = idxx,column = model_col).value)] for idxx in range(2, duke_all.max_row + 1)]
	duke_names_with_dx_model_feat = [[ii[0],0,ii[2]] if 'pure' in ii[1] else [ii[0],1,ii[2]] for ii in duke_names_with_dx_model_feat_ori]
    ########
	duke_id_diagnosis_model_feats = []
	for one_info in (duke_names_with_dx_model_feat):
		one_feat = [fea for fea in duke_feature_paths_all if one_info[0] in fea]
		duke_id_diagnosis_model_feats.append(one_info + one_feat)

	X_hologic_duke = np.array([]).reshape(0,num_radiomic_features)
	Y_hologic_duke = np.array([])
	X_ge_duke = np.array([]).reshape(0,num_radiomic_features)
	Y_ge_duke = np.array([])
	info_duke_ge = []
	info_duke_hologic = []
	for one_id, one_dx, one_model, one_added_feat in duke_id_diagnosis_model_feats:
		X_one_feat = np.load(one_added_feat).reshape((1,num_radiomic_features))
		if 'Senograph' in one_model:
			X_ge_duke = np.concatenate((X_ge_duke,X_one_feat),axis = 0)
			Y_ge_duke = np.concatenate((Y_ge_duke, [one_dx]))
			info_duke_ge.append([one_id, one_dx, one_model, one_added_feat])
		if 'Selenia' in one_model:
			X_hologic_duke = np.concatenate((X_hologic_duke,X_one_feat),axis = 0)
			Y_hologic_duke = np.concatenate((Y_hologic_duke,[one_dx]))
			info_duke_hologic.append([one_id, one_dx, one_model, one_added_feat])
	#######
	[X_ge_duke_normed,_] = mean_std_normalize(X_ge_duke, [])
	[X_hologic_duke_normed,_] = mean_std_normalize(X_hologic_duke, [])
	X_duke_normed = np.concatenate((X_ge_duke_normed, X_hologic_duke_normed), axis= 0)
	Y_duke = np.concatenate((Y_ge_duke, Y_hologic_duke), axis = 0)
	info_duke = info_duke_ge + info_duke_hologic
	################################## Netherlands Cases ################################
	nki_xlsx_file_name = 'NKI/NL-NKI information.xlsx'
	nki_file = openpyxl.load_workbook(nki_xlsx_file_name)
	nki_image_sheet = nki_file.worksheets[0]
	nki_image_item_list = [str(nki_image_sheet.cell(row = 1, column = col_now).value) for col_now in range(1, nki_image_sheet.max_column + 1)]
	nki_caseid_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'new_id' in ii][0]
	nki_diagnosis_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'upgrade' in ii][0]
	nki_model_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'Manufacturers Model Name' in ii][0]
	nki_age_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'age' in ii][0]
	nki_caseid_and_diagnosis_model_lesion_age = [[str(nki_image_sheet.cell(row = idds, column = nki_caseid_idx).value), \
		str(nki_image_sheet.cell(row = idds, column = nki_diagnosis_idx).value), \
		str(nki_image_sheet.cell(row = idds, column = nki_model_idx).value), \
		str(nki_image_sheet.cell(row = idds, column = nki_lesion_idx).value), float(nki_image_sheet.cell(row = idds, column = nki_age_idx).value)] \
		for idds in range(2,nki_image_sheet.max_row+1)]
	nki_feats_all = glob.glob('NKI/features_npy/*.npy')
	nki_diagnosis_id_model_lesion_age_imagename = []
	for one_feat in nki_feats_all:
		nki_diagnosis_id_model_lesion_age_imagename.append([ss + [one_feat] for ss in nki_caseid_and_diagnosis_model_lesion_age if ss[0] in one_feat][0])
	####
	X_hologic_nki = np.array([]).reshape(0, num_radiomic_features)
	Y_hologic_nki = np.array([]).astype(np.uint8)
	info_nki_hologic = []
	for one_id, one_dx, one_model, one_lesion, one_age, one_added_feat in nki_diagnosis_id_model_lesion_age_imagename:
		X_one_feat = np.load(one_added_feat).reshape((1,num_radiomic_features))
		if 'Selenia' in one_model:
			X_hologic_nki = np.concatenate((X_hologic_nki,X_one_feat),axis = 0)
			Y_hologic_nki = np.concatenate((Y_hologic_nki,[1 if 'Upstage' in one_dx else 0]))
			info_nki_hologic.append([one_id, one_added_feat.split('/')[-1].split('_feat')[0], 1 if 'Upstage' in one_dx else 0, one_model, one_added_feat])
	################################## UK Cases ################################
	optimam_xlsx_file_name = 'OPTIMAM/UK-OPTIMAM information.xlsx'
	optimam_file = openpyxl.load_workbook(optimam_xlsx_file_name)
	optimam_image_sheet = optimam_file.worksheets[0]
	optimam_image_item_list = [str(optimam_image_sheet.cell(row = 1, column = col_now).value) for col_now in range(1, optimam_image_sheet.max_column + 1)]
	optimam_caseid_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'ClientID' in ii][0]
	optimam_imageid_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'Image Name' in ii][0]
	optimam_diagnosis_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'Diagnosis' in ii][0]
	optimam_model_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'Manufacturers Model Name' in ii][0]
	optimam_caseid_image_diagnosis_model = [[str(optimam_image_sheet.cell(row = idds, column = optimam_caseid_idx).value), \
			str(optimam_image_sheet.cell(row = idds, column = optimam_imageid_idx).value), \
			str(optimam_image_sheet.cell(row = idds, column = optimam_diagnosis_idx).value), \
			str(optimam_image_sheet.cell(row = idds, column = optimam_model_idx).value)] for idds in range(2, optimam_image_sheet.max_row+1)]
	optimam_features_all = glob.glob(optimam_features_root + 'features_npy/*.npy')
	optimam_caseid_image_diagnosis_model_with_feats = []
	for one_feat in optimam_features_all:
		optimam_caseid_image_diagnosis_model_with_feats.append([ss + [one_feat] for ss in optimam_caseid_image_diagnosis_model if ss[1] in one_feat][0])
	####
	X_ge_optimam = np.array([]).reshape(0,num_radiomic_features)
	Y_ge_optimam = np.array([]).astype(np.uint8)
	X_hologic_optimam = np.array([]).reshape(0,num_radiomic_features)
	Y_hologic_optimam = np.array([]).astype(np.uint8)
	info_optimam_hologic = []
	info_optimam_ge = []
	####
	for one_id, one_image, one_dx, one_model, one_added_feat in optimam_caseid_image_diagnosis_model_with_feats:
		X_one_feat = np.load(one_added_feat).reshape((1,num_radiomic_features))
		if 'Selenia' in one_model:
			X_hologic_optimam = np.concatenate((X_hologic_optimam,X_one_feat),axis = 0)
			Y_hologic_optimam = np.concatenate((Y_hologic_optimam,[0 if 'pure' in one_dx.lower() else 1]))
			info_optimam_hologic.append([one_id, one_image, 0 if 'pure' in one_dx.lower() else 1, one_model, one_added_feat])
		if 'Senograph' in one_model:
			X_ge_optimam = np.concatenate((X_ge_optimam,X_one_feat),axis = 0)
			Y_ge_optimam = np.concatenate((Y_ge_optimam,[0 if 'pure' in one_dx.lower() else 1]))
			info_optimam_ge.append([one_id, one_image, 0 if 'pure' in one_dx.lower() else 1, one_model, one_added_feat])
	#######################################
	#######################################
	f_log = open(path_to_save_train_logs + 'logs_CV_ONLY_LogisticRegression_WITH_and_WITHOUT_feature_selection_with_saving_predictions.log','a')
	##
	num_training_pure = np.where(Y_duke==0)[0].shape[0]
	num_training_upstaging = np.where(Y_duke==1)[0].shape[0]
	num_holdout_pure = int(round(num_training_pure/float(nfold)))
	num_holdout_upstaging = int(round(num_training_upstaging/float(nfold)))
	##
	f_csv_cv_labels = open(path_to_save_train_logs + 'CV_ONLY_Labels_with_%dfolds_%diterations.csv'%(nfold,iterTimes),'w')
	wri_cv_labels = csv.writer(f_csv_cv_labels)
	##
	best_c_train_duke_all = []
	auc_train_duke_all_nofs = np.zeros([iterTimes,nfold])
	f_csv_cv_nofs = open(path_to_save_train_logs + 'CV_ONLY_NO-FS_Predictions_with_%dfolds_%diterations.csv'%(nfold,iterTimes),'w')
	wri_cv_nofs = csv.writer(f_csv_cv_nofs)
	##
	best_c_train_duke_all_with_fs = []
	auc_train_duke_all_with_fs = np.zeros([iterTimes,nfold])
	f_csv_cv_with_fs = open(path_to_save_train_logs + 'CV_ONLY_WITH-FS_Predictions_with_%dfolds_%diterations.csv'%(nfold,iterTimes),'w')
	wri_cv_with_fs = csv.writer(f_csv_cv_with_fs)
	##
	idx_feat_picked_train_duke_all = []
	feat_picked_train_duke_all = []
	ind0 = [indsub for indsub,x in enumerate(Y_duke) if x==0]
	ind1 = [indsub for indsub,x in enumerate(Y_duke) if x==1]
	for iter2,seed_val in zip(range(iterTimes),range(iterTimes)):	
		random.seed((iterTimes-seed_val)*2);random.shuffle(ind0)	
		random.seed((iterTimes-seed_val)*2);random.shuffle(ind1)
		for nf in range(nfold):
			ind_val = ind0[nf*num_holdout_pure:nf*num_holdout_pure+num_holdout_pure]+ind1[nf*num_holdout_upstaging:nf*num_holdout_upstaging+num_holdout_upstaging]
			ind_tr = list(set(ind0+ind1)-set(ind_val))
			X_new_tr = np.copy(X_duke_normed[np.array(ind_tr),:])
			Y_tr = np.array(Y_duke[np.array(ind_tr)])
			X_new_val = np.copy(X_duke_normed[np.array(ind_val),:])
			Y_val = np.array(Y_duke[np.array(ind_val)])
			wri_cv_labels.writerow(Y_val.tolist())
			X_tr = np.copy(X_new_tr);X_val = np.copy(X_new_val)
			lr_baseline_infold = GridSearchCV(linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 20000), param_grid, cv=embedded_cv,scoring='roc_auc', n_jobs = num_cpus_to_load)
			_=lr_baseline_infold.fit(X_tr,Y_tr)
			best_c_now = lr_baseline_infold.best_params_['C']
			best_c_train_duke_all.extend([best_c_now])
			val_proba_now = lr_baseline_infold.predict_proba(X_val)[:,1]
			wri_cv_nofs.writerow(val_proba_now.tolist())
			auc_train_duke_all_nofs[iter2,nf] = metrics.roc_auc_score(Y_val,val_proba_now)
			##
			randomized_logistic = RandomizedLogisticRegression(C=1,random_state=nfold)
			randomized_logistic.fit(X_tr,Y_tr)
			ind_fea = randomized_logistic.all_scores_[:,0]
			rank_one_ind_fea = np.flipud(np.argsort(ind_fea))
			current_idx_feat_picked = [i for i in rank_one_ind_fea if ind_fea[i]>0]
			if len(current_idx_feat_picked) > 0:
				current_feat_picked = np.array(radiomics_feat_list)[np.array(current_idx_feat_picked)]
				idx_feat_picked_train_duke_all.append(current_idx_feat_picked)
				feat_picked_train_duke_all.append(current_feat_picked)
				X_tr = np.copy(X_new_tr[:,np.array(current_idx_feat_picked)])
				X_val = np.copy(X_new_val[:,np.array(current_idx_feat_picked)])
				lr_train_duke_all_infold_with_fs = GridSearchCV(linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 20000), param_grid, cv=embedded_cv,scoring='roc_auc', n_jobs = num_cpus_to_load)
				_=lr_train_duke_all_infold_with_fs.fit(X_tr,Y_tr)
				best_c_now = lr_train_duke_all_infold_with_fs.best_params_['C']
				best_c_train_duke_all_with_fs.extend([best_c_now])
				val_proba_now_with_fs = lr_train_duke_all_infold_with_fs.predict_proba(X_val)[:,1]
				wri_cv_with_fs.writerow(val_proba_now_with_fs.tolist())
				auc_train_duke_all_with_fs[iter2,nf] = metrics.roc_auc_score(Y_val,val_proba_now_with_fs)
			else:
				auc_train_duke_all_with_fs[iter2,nf] = 0.5
		print('---------Radiomics Logistic Regression Run %d NO fs AUC: %0.3f +- %0.3f; WITH fs AUC: %0.3f +- %0.3f' \
			%(iter2+1, round(auc_train_duke_all_nofs[:iter2+1].mean(),3),round(auc_train_duke_all_nofs[:iter2+1].mean(1).std(),3), \
				round(auc_train_duke_all_with_fs[:iter2+1].mean(),3),round(auc_train_duke_all_with_fs[:iter2+1].mean(1).std(),3)) )
		f_log.write('---------Radiomics Logistic Regression Run %d NO fs AUC: %0.3f +- %0.3f; WITH fs AUC: %0.3f +- %0.3f\n' \
			%(iter2+1, round(auc_train_duke_all_nofs[:iter2+1].mean(),3),round(auc_train_duke_all_nofs[:iter2+1].mean(1).std(),3), \
				round(auc_train_duke_all_with_fs[:iter2+1].mean(),3),round(auc_train_duke_all_with_fs[:iter2+1].mean(1).std(),3)) )

	np.save(path_to_save_train_logs+'CV_ONLY_Training_NO_FS_LogisticRegression_c.npy',best_c_train_duke_all)
	np.save(path_to_save_train_logs+'CV_ONLY_Training_WITH_FS_LogisticRegression_c.npy',best_c_train_duke_all_with_fs)
	np.save(path_to_save_train_logs+'CV_ONLY_Training_WITH_FS_LogisticRegression_picked_features_idx.npy',idx_feat_picked_train_duke_all)

	f_feat_picked = open(path_to_save_train_logs + 'CV_ONLY_Training_WITH_FS_LogisticRegression_picked_features_names.txt','w')
	for one_select in feat_picked_train_duke_all:
		f_feat_picked.write(' + '.join(one_select)+'\n')
	f_feat_picked.close()
	f_log.close()
	f_csv_cv_labels.close()
	f_csv_cv_nofs.close()
	f_csv_cv_with_fs.close()

	#########################CV Top Features Performance###############
	selected_features_num = 11
	f_log = open(path_to_save_train_logs + 'logs_CV_ONLY_selected_TOP_%dFeatures_saving_predictions.log'%(selected_features_num),'a')
	f_csv_cv_labels = open(path_to_save_train_logs + 'With_%dTOP_features_from_CV_feature_selection_labels.csv'%(selected_features_num),'w')
	wri_cv_labels = csv.writer(f_csv_cv_labels)
	best_c_train_duke_all_with_fs = []
	auc_train_duke_all_top_feats = np.zeros([iterTimes,nfold])
	f_csv_cv_with_fs = open(path_to_save_train_logs + 'With_%dTOP_features_from_CV_feature_selection_predictions.csv'%(selected_features_num),'w')
	wri_cv_with_fs = csv.writer(f_csv_cv_with_fs)
	##
	idx_feat_picked_train_duke_all = np.load(path_to_save_train_logs+'CV_ONLY_Training_WITH_FS_LogisticRegression_picked_features_idx.npy', allow_pickle = True)
	uni_feat,counts_feat = np.unique([one_feat for one_fold_feats in idx_feat_picked_train_duke_all.tolist() for one_feat in one_fold_feats],return_counts = True)
	sorted_feat = uni_feat[np.flipud(np.argsort(counts_feat))]
	sorted_counts = np.flipud(np.sort(counts_feat))

	X_duke_train_subset = np.copy(X_duke_normed[:,sorted_feat[:selected_features_num]])
	current_feat_picked = np.array(radiomics_feat_list)[sorted_feat[:selected_features_num]].tolist()
	##
	num_training_pure = np.where(Y_duke==0)[0].shape[0]
	num_training_upstaging = np.where(Y_duke==1)[0].shape[0]
	num_holdout_pure = int(round(num_training_pure/float(nfold)))
	num_holdout_upstaging = int(round(num_training_upstaging/float(nfold)))
	ind0 = [indsub for indsub,x in enumerate(Y_duke) if x==0]
	ind1 = [indsub for indsub,x in enumerate(Y_duke) if x==1]
	for iter2,seed_val in zip(range(iterTimes),range(iterTimes)):	
		random.seed((iterTimes-seed_val)*2);random.shuffle(ind0)	
		random.seed((iterTimes-seed_val)*2);random.shuffle(ind1)
		for nf in range(nfold):
			ind_val = ind0[nf*num_holdout_pure:nf*num_holdout_pure+num_holdout_pure]+ind1[nf*num_holdout_upstaging:nf*num_holdout_upstaging+num_holdout_upstaging]
			ind_tr = list(set(ind0+ind1)-set(ind_val))
			X_new_tr = np.copy(X_duke_train_subset[np.array(ind_tr),:])
			Y_tr = np.array(Y_duke[np.array(ind_tr)])
			X_new_val = np.copy(X_duke_train_subset[np.array(ind_val),:])
			Y_val = np.array(Y_duke[np.array(ind_val)])
			wri_cv_labels.writerow(Y_val.tolist())
			X_tr = np.copy(X_new_tr);X_val = np.copy(X_new_val)
			lr_baseline_infold_with_fs = GridSearchCV(linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 10000), param_grid, cv=embedded_cv,scoring='roc_auc', n_jobs = num_cpus_to_load)
			_=lr_baseline_infold_with_fs.fit(X_tr,Y_tr)
			best_c_now = lr_baseline_infold_with_fs.best_params_['C']
			best_c_train_duke_all_with_fs.extend([best_c_now])
			val_proba_now_with_fs = lr_baseline_infold_with_fs.predict_proba(X_val)[:,1]
			wri_cv_with_fs.writerow(val_proba_now_with_fs.tolist())
			auc_train_duke_all_top_feats[iter2,nf] = metrics.roc_auc_score(Y_val,val_proba_now_with_fs)
		print('--------- Feature Num: %d, Logistic Regression Run %d  auc: %0.3f +- %0.3f' %(selected_features_num,iter2+1, round(auc_train_duke_all_top_feats[:iter2+1].mean(),3),round(auc_train_duke_all_top_feats[:iter2+1].mean(1).std(),3)))
		f_log.write('--------- Feature Num: %d, Logistic Regression Run %d auc: %0.3f +- %0.3f\n' %(selected_features_num, iter2+1, round(auc_train_duke_all_top_feats[:iter2+1].mean(),3),round(auc_train_duke_all_top_feats[:iter2+1].mean(1).std(),3)))
	print('---- %d Number of TOP Features, Logistic Regression auc: %0.3f +- %0.3f, with Features: %s' %(selected_features_num, round(auc_train_duke_all_top_feats.mean(),3),round(auc_train_duke_all_top_feats.mean(1).std(),3),current_feat_picked))
	f_log.write('---- %d Number of TOP Features, Logistic Regression auc: %0.3f +- %0.3f, with Features: %s\n' %(selected_features_num, round(auc_train_duke_all_top_feats.mean(),3),round(auc_train_duke_all_top_feats.mean(1).std(),3),current_feat_picked))
	##
	np.save(path_to_save_train_logs + 'With_%dTOP_features_from_CV_feature_selection_best_parameters_C.npy'%(selected_features_num),best_c_train_duke_all_with_fs)
	f_csv_cv_labels.close()
	f_csv_cv_with_fs.close()
	
	########################################################################################
	########################################################################################
	selected_radiomics_features_num = 11
	## DCIS-R
	best_c_train_on_all = np.load( 'training_results/CV_ONLY_Training_NO_FS_LogisticRegression_c.npy', allow_pickle = True)
	uni_c_radiomics_nofs,counts_c_radiomics_nofs = np.unique(best_c_train_on_all,return_counts = True)
	most_selected_c_radiomics_nofs = uni_c_radiomics_nofs[np.flipud(np.argsort(counts_c_radiomics_nofs))][0]
	lr_radiomics_nofs = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 10000,C = most_selected_c_radiomics_nofs)
	_=lr_radiomics_nofs.fit(X_duke_normed, Y_duke)
	## DCIS-Rs
	idx_feat_picked_train_on_all = np.load('training_results/CV_ONLY_Training_WITH_FS_LogisticRegression_picked_features_idx.npy', allow_pickle = True).tolist()
	uni_radiomics_feats, counts_radiomics_feats = np.unique([item for sublist in idx_feat_picked_train_on_all for item in sublist], return_counts = True)
	selected_feature_idx = uni_radiomics_feats[np.flipud(np.argsort(counts_radiomics_feats))][:selected_radiomics_features_num]
	X_duke_normed_subset = np.copy(X_duke_normed[:, selected_feature_idx])
	selected_feature_names = np.array(radiomics_feat_list)[selected_feature_idx]
	#####
	best_c_train_duke_all_top_features = np.load( 'training_results/With_%dTOP_features_from_CV_feature_selection_best_parameters_C.npy'%(selected_radiomics_features_num))
	uni_c_train_duke_all_topfeats, counts_c_train_duke_all_topfeats = np.unique(best_c_train_duke_all_top_features,return_counts = True)
	most_selected_c_train_duke_all_top_feats = uni_c_train_duke_all_topfeats[np.flipud(np.argsort(counts_c_train_duke_all_topfeats))][0]
	lr_radiomics_topfeats = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 10000,C = most_selected_c_train_duke_all_top_feats)
	_=lr_radiomics_topfeats.fit(X_duke_normed_subset,Y_duke)
	##
	##################################################################
	##################################################################
	#### AUC Performance On OPTIMAM
	[X_hologic_optimam_normed, _] = mean_std_normalize(X_hologic_optimam, [])
	[X_ge_optimam_normed, _] = mean_std_normalize(X_ge_optimam, [])
	X_optimam_normed = np.concatenate((X_hologic_optimam_normed, X_ge_optimam_normed), axis= 0)
	Y_optimam = np.concatenate((Y_hologic_optimam, Y_ge_optimam), axis= 0)
	info_optimam = info_optimam_hologic + info_optimam_ge
	X_optimam_normed_subset = np.copy(X_optimam_normed[:, selected_feature_idx])
	proba_optimam_topfeats = lr_radiomics_topfeats.predict_proba(X_optimam_normed_subset)[:,1]
	auc_optimam_topfeats = metrics.roc_auc_score(Y_optimam,proba_optimam_topfeats)
	## Case AUC:
	optimam_unique_cases = np.unique([one_list[0] for one_list in info_optimam])
	Y_optimam_by_case = []
	proba_optimam_topfeats_by_case_mean = []
	info_optimam_by_case = []
	for one_case in optimam_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam) if one_list[0]==one_case]
		info_optimam_by_case.append([one_list for one_list in info_optimam if one_list[0]==one_case])
		proba_optimam_topfeats_by_case_mean.append(np.mean([proba_optimam_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_by_case.append([info_optimam[ids][2] for ids in one_case_ids][0])
	auc_optimam_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_topfeats_by_case_mean))
	####
	with open('training_results/Train_on_us_test_on_uk.csv','w') as f_csv_optimam:
		wri_csv_optimam = csv.writer(f_csv_optimam)
		_= wri_csv_optimam.writerow(['ID','labels','pred.top.feats'])
		_=[wri_csv_optimam.writerow([info_optimam_by_case[idss][0], Y_optimam_by_case[idss], \
			proba_optimam_topfeats_by_case_mean[idss]]) for idss in range(Y_optimam_by_case.shape[0])]
	##################################################################
	##################################################################
	#### AUC Performance On NKI
	[X_hologic_nki_normed,_] = mean_std_normalize(X_hologic_nki, [])
	proba_nki_topfeats_hologic_only = lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1]
	auc_nki_topfeats_hologic_only = metrics.roc_auc_score(Y_hologic_nki, proba_nki_topfeats_hologic_only)
	##
	nki_hologic_unique_cases = np.unique([one_list[0] for one_list in info_nki_hologic])
	Y_nki_hologic_by_case = []
	proba_nki_hologic_topfeats_by_case_mean = []
	info_nki_hologic_by_case = []
	for one_case in nki_hologic_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_nki_hologic) if one_list[0]==one_case]
		info_nki_hologic_by_case.append([one_list in info_nki_hologic if one_list[0]==one_case]) 
		proba_nki_hologic_topfeats_by_case_mean.append(np.mean([proba_nki_topfeats_hologic_only[ids] for ids in one_case_ids]))
		Y_nki_hologic_by_case.append([info_nki_hologic[ids][2] for ids in one_case_ids][0])

	auc_nki_hologic_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_topfeats_by_case_mean))
	with open('training_results/Train_on_us_test_on_nl.csv','w') as f_csv_nki_hologic:
		wri_csv_nki_hologic = csv.writer(f_csv_nki_hologic)
		_= wri_csv_nki_hologic.writerow(['ID','labels','pred.top.feats'])
		_=[wri_csv_nki_hologic.writerow([info_nki_hologic_by_case[idss][0], Y_nki_hologic_by_case[idss], proba_nki_hologic_topfeats_by_case_mean]) for idss in range(Y_nki_hologic_by_case.shape[0])]
	############################################################
