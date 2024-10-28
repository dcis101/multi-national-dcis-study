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
	path_to_save_train_logs = 'training_results/results_norm_independently/Duke_all_data_cross-validation_with_fs_py2/'
	########################################DUKE CASES **** DUKE CASES **** DUKE CASES###################################
	duke_features_root = 'DUKE/'
	nki_features_root ='NKI/'
	optimam_features_root = 'OPTIMAM/'
	####
	cluster_features_names = ['MCC Area', 'MCC Eccentricity', 'MCC NO of MCs',  'MCC MCs Coverage', 'MCC Background Mean', 'MCC Background Std', 'MCC All MCs Intensity Mean', 'MCC All MCs Intensity Std', 'Mean GLCM Contrast', 'Mean GLCM Correlation', 'Mean GLCM Energy', 'Mean GLCM Homogeneity', 'Lesion Size']
	ind_features_names = ['MC Perimeter', 'MC Area', 'MC Circularity', 'MC Eccentricity', 'MC Major Axis', 'MC Minor Axis', '1st Variant of Hu Moments','2nd Variant of Hu Moments','3rd Variant of Hu Moments','4th Variant of Hu Moments','5th Variant of Hu Moments','6th Variant of Hu Moments','7th Variant of Hu Moments','MC Distance to Cluster Centroid','MC Distance to Nearest MC','MC Normalized Degree','MC Intensity Mean','MC Intensity Std','MC Background Intensity Mean','MC Background Intensity Std','MC Mean GLCM Contrast','MC Mean GLCM Correlation','MC Mean GLCM Energy','MC Mean GLCM Homogeneity']
	radiomics_feat_list =	cluster_features_names[:-1] + [' '.join(['Mean',ss]) for ss in ind_features_names] + [' '.join(['Std',ss]) for ss in ind_features_names] + \
						[' '.join(['Min',ss]) for ss in ind_features_names] + [' '.join(['Max',ss]) for ss in ind_features_names] + \
						cluster_features_names[-1:]

	num_radiomic_features = len(radiomics_feat_list)
	##***
	duke_xlsx_file_name_alldcis = 'Information_Files/Finalized_DCIS_v1.0-5.0_DCIS700_ADHIDC230_with_Clinical_Features_train400_test300_2019Nov26.xlsx'
	dcis_file = openpyxl.load_workbook(duke_xlsx_file_name_alldcis)
	dcis_list_all = dcis_file.worksheets[0]
	dcis_training= dcis_file.worksheets[1]
	dcis_testing = dcis_file.worksheets[2]
	#########DCIS
	npy_dcis_path_all_ge_nonge = []
	ver_paths_all = ['DUKE']
	vars()['testing_dcis_path_v' + str(0)] = duke_features_root  + ver_paths_all[0] + 'Feature109_Final_with_unet_ver11-ep260/'
	vars()['npy_testing_dcis_path_v' + str(version)] = glob.glob(vars()['testing_dcis_path_v' + str(version)]+'*.npy')
	npy_dcis_path_all_ge_nonge += vars()['npy_testing_dcis_path_v' + str(version)]

	#################### Training Cases ########################################
	item_list = [str(dcis_training.cell(row = 1, column = col_now).value) for col_now in range(1, dcis_training.max_column + 1)]
	dx_col = [col_now + 1 for col_now, ii in enumerate(item_list) if 'Diagnosis' in ii][0]
	model_col = [col_now + 1 for col_now, ii in enumerate (item_list) if 'Manufacturer Model Name' in ii][0]
	fakeid_col = [col_now + 1 for col_now, ii in enumerate(item_list) if 'Fake ID' in ii][0]
	##
	training_names_with_dx_model_cli_feat = [[str(dcis_training.cell(row = idxx,column = fakeid_col).value),str(dcis_training.cell(row = idxx,column = dx_col).value),str(dcis_training.cell(row = idxx,column = model_col).value)] for idxx in range(2, dcis_training.max_row + 1)]
	training_names_with_dx_model_cli_feat = [[ii[0],0,ii[2]] if 'pure' in ii[1] else [ii[0],1,ii[2]] for ii in training_names_with_dx_model_cli_feat]
	################# Rest Testing Cases ##################################
	item_list = [str(dcis_testing.cell(row = 1, column = col_now).value) for col_now in range(1, dcis_testing.max_column + 1)]
	dx_col = [col_now + 1 for col_now, ii in enumerate(item_list) if 'Diagnosis' in ii][0]
	model_col = [col_now + 1 for col_now, ii in enumerate(item_list) if 'Manufacturer Model Name' in ii][0]
	fakeid_col = [col_now + 1 for col_now, ii in enumerate(item_list) if 'Fake ID' in ii][0]
	##
	testing_names_with_dx_model_cli_feat = [[str(dcis_testing.cell(row = idxx,column = fakeid_col).value),str(dcis_testing.cell(row = idxx,column = dx_col).value),str(dcis_testing.cell(row = idxx,column = model_col).value)] for idxx in range(2, dcis_testing.max_row + 1)]
	testing_names_with_dx_model_cli_feat= [[ii[0],0,ii[2]] if 'pure' in ii[1] else [ii[0],1,ii[2]] for ii in testing_names_with_dx_model_cli_feat]
	########################################################################################
	duke_id_diagnosis_model_feats = []
	for one_info in (training_names_with_dx_model_cli_feat + testing_names_with_dx_model_cli_feat):
		one_feat = [fea for fea in npy_dcis_path_all_ge_nonge if one_info[0] in fea]
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

	############################# Normalize Normalize Normalize ##########################
	## CLASSIC CR, Clearview CSm, Clearview CSm, Mammomat Novation DR removed
	[X_ge_duke_normed,_] = mean_std_normalize(X_ge_duke, [])
	[X_hologic_duke_normed,_] = mean_std_normalize(X_hologic_duke, [])
	X_duke_normed = np.concatenate((X_ge_duke_normed, X_hologic_duke_normed), axis= 0)
	Y_duke = np.concatenate((Y_ge_duke, Y_hologic_duke), axis = 0)
	info_duke = info_duke_ge + info_duke_hologic
	########################################################################################################################
	
	f_log = open(path_to_save_train_logs + 'logs_CV_ONLY_LogisticRegression_WITH_and_WITHOUT_feature_selection_with_saving_predictions.log','a')
	## Model A on 400 Training Data with Logistic Regression
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

	np.save(path_to_save_train_logs+'CV_ONLY_Training_NO_FS_400_LogisticRegression_c.npy',best_c_train_duke_all)
	np.save(path_to_save_train_logs+'CV_ONLY_Training_WITH_FS_400_LogisticRegression_c.npy',best_c_train_duke_all_with_fs)
	np.save(path_to_save_train_logs+'CV_ONLY_Training_WITH_FS_400_LogisticRegression_picked_features_idx.npy',idx_feat_picked_train_duke_all)

	f_feat_picked = open(path_to_save_train_logs + 'CV_ONLY_Training_WITH_FS_400_LogisticRegression_picked_features_names.txt','w')
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
	idx_feat_picked_train_duke_all = np.load(path_to_save_train_logs+'CV_ONLY_Training_WITH_FS_400_LogisticRegression_picked_features_idx.npy', allow_pickle = True)
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
	nki_xlsx_file_name = 'NKI/NKI_Mammo_For_Duke/NKI_cases_diagnosis.xlsx'
	nki_file = openpyxl.load_workbook(nki_xlsx_file_name)
	nki_image_sheet = nki_file.worksheets[0]
	nki_image_item_list = [str(nki_image_sheet.cell(row = 1, column = col_now).value) for col_now in range(1, nki_image_sheet.max_column + 1)]
	nki_caseid_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'new_id' in ii][0]
	nki_diagnosis_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'upgrade' in ii][0]
	nki_model_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'Manufacturers Model Name' in ii][0]
	nki_lesion_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'mammo_lesion_type' in ii][0]
	nki_age_idx = [col_now + 1 for col_now, ii in enumerate(nki_image_item_list) if 'age' in ii][0]
	nki_caseid_and_diagnosis_model_lesion_age = [[str(nki_image_sheet.cell(row = idds, column = nki_caseid_idx).value), \
		str(nki_image_sheet.cell(row = idds, column = nki_diagnosis_idx).value), \
		str(nki_image_sheet.cell(row = idds, column = nki_model_idx).value), \
		str(nki_image_sheet.cell(row = idds, column = nki_lesion_idx).value), float(nki_image_sheet.cell(row = idds, column = nki_age_idx).value)] \
		for idds in range(2,nki_image_sheet.max_row+1)]
	nki_feats_all = glob.glob(nki_features_root + 'Feature109_Final_with_unet_ver11-ep260/*.npy')
	nki_diagnosis_id_model_lesion_age_imagename = []
	for one_feat in nki_feats_all:
		nki_diagnosis_id_model_lesion_age_imagename.append([ss + [one_feat] for ss in nki_caseid_and_diagnosis_model_lesion_age if ss[0] in one_feat][0])


	################################################################################################################
	X_ge_nki = np.array([]).reshape(0, num_radiomic_features)
	Y_ge_nki = np.array([]).astype(np.uint8)
	X_hologic_nki = np.array([]).reshape(0, num_radiomic_features)
	Y_hologic_nki = np.array([]).astype(np.uint8)
	X_siemens_nki = np.array([]).reshape(0, num_radiomic_features)
	Y_siemens_nki = np.array([]).astype(np.uint8)
	info_nki_ge = []
	info_nki_hologic = []
	info_nki_siemens = []
	for one_id, one_dx, one_model, one_lesion, one_age, one_added_feat in nki_diagnosis_id_model_lesion_age_imagename:
		# if (('mass' not in one_lesion.lower()) and ('architectural' not in one_lesion.lower()) and ('microcalcification' in one_lesion.lower()) and (one_age>=40)):
		if (('mass' not in one_lesion.lower()) and ('architectural' not in one_lesion.lower()) and ('microcalcification' in one_lesion.lower())):
			X_one_feat = np.load(one_added_feat).reshape((1,num_radiomic_features))
			if 'Senograph' in one_model:
				X_ge_nki = np.concatenate((X_ge_nki,X_one_feat),axis = 0)
				Y_ge_nki = np.concatenate((Y_ge_nki, [1 if 'Upstage to IBC' in one_dx else 0]))
				info_nki_ge.append([one_id, one_added_feat.split('/')[-1].split('_feat')[0], 1 if 'Upstage to IBC' in one_dx else 0, one_model, one_added_feat])
			if 'Selenia' in one_model:
				X_hologic_nki = np.concatenate((X_hologic_nki,X_one_feat),axis = 0)
				Y_hologic_nki = np.concatenate((Y_hologic_nki,[1 if 'Upstage to IBC' in one_dx else 0]))
				info_nki_hologic.append([one_id, one_added_feat.split('/')[-1].split('_feat')[0], 1 if 'Upstage to IBC' in one_dx else 0, one_model, one_added_feat])
			if 'Mammo' in one_model:
				X_siemens_nki = np.concatenate((X_siemens_nki,X_one_feat),axis = 0)
				Y_siemens_nki = np.concatenate((Y_siemens_nki,[1 if 'Upstage to IBC' in one_dx else 0]))
				info_nki_siemens.append([one_id, one_added_feat.split('/')[-1].split('_feat')[0], 1 if 'Upstage to IBC' in one_dx else 0, one_model, one_added_feat])

	####################################################################
	optimam_xlsx_file_name = '/home/rui/Documents/OPTIMAM/OPTIMAM_DCIS_lesion_with_model_info.xlsx'
	optimam_file = openpyxl.load_workbook(optimam_xlsx_file_name)
	optimam_image_sheet = optimam_file.worksheets[0]
	optimam_image_item_list = [str(optimam_image_sheet.cell(row = 1, column = col_now).value) for col_now in range(1, optimam_image_sheet.max_column + 1)]
	optimam_caseid_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'ClientID' in ii][0]
	optimam_imageid_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'IMAGESOPUID' in ii][0]
	optimam_diagnosis_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'Diagnosis' in ii][0]
	optimam_model_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'Manufacturers Model Name' in ii][0]
	optimam_faint_idx = [col_now + 1 for col_now, ii in enumerate(optimam_image_item_list) if 'Conspicuity' in ii][0]
	optimam_caseid_image_diagnosis_model = [[str(optimam_image_sheet.cell(row = idds, column = optimam_caseid_idx).value), \
			str(optimam_image_sheet.cell(row = idds, column = optimam_imageid_idx).value), \
			str(optimam_image_sheet.cell(row = idds, column = optimam_diagnosis_idx).value), \
			str(optimam_image_sheet.cell(row = idds, column = optimam_model_idx).value)] for idds in range(2, optimam_image_sheet.max_row+1)]
	optimam_features_all = glob.glob(optimam_features_root + 'Feature109_Final_with_unet_ver11-ep260/*.npy')
	optimam_caseid_image_diagnosis_model_with_feats = []
	for one_feat in optimam_features_all:
		optimam_caseid_image_diagnosis_model_with_feats.append([ss + [one_feat] for ss in optimam_caseid_image_diagnosis_model if ss[1] in one_feat][0])

	##
	X_ge_optimam = np.array([]).reshape(0,num_radiomic_features)
	Y_ge_optimam = np.array([]).astype(np.uint8)
	X_hologic_optimam = np.array([]).reshape(0,num_radiomic_features)
	Y_hologic_optimam = np.array([]).astype(np.uint8)
	X_philips_optimam = np.array([]).reshape(0,num_radiomic_features)
	Y_philips_optimam = np.array([]).astype(np.uint8)
	X_siemens_optimam = np.array([]).reshape(0,num_radiomic_features)
	Y_siemens_optimam = np.array([]).astype(np.uint8)
	info_optimam_hologic = []
	info_optimam_ge = []
	info_optimam_philips = []
	info_optimam_siemens = []

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
		if (('L30' in one_model) or ('MammoDiagnost DR' in one_model)):
			X_philips_optimam = np.concatenate((X_philips_optimam,X_one_feat),axis = 0)
			Y_philips_optimam = np.concatenate((Y_philips_optimam,[0 if 'pure' in one_dx.lower() else 1]))
			info_optimam_philips.append([one_id, one_image, 0 if 'pure' in one_dx.lower() else 1, one_model, one_added_feat])
		if 'Mammomat' in one_model:
			X_siemens_optimam = np.concatenate((X_siemens_optimam,X_one_feat),axis = 0)
			Y_siemens_optimam = np.concatenate((Y_siemens_optimam,[0 if 'pure' in one_dx.lower() else 1]))
			info_optimam_siemens.append([one_id, one_image, 0 if 'pure' in one_dx.lower() else 1, one_model, one_added_feat])

	##############################################################################
	##############################################################################

	selected_radiomics_features_num = 11
	duke_train_on_all_path  = 'training_results/results_norm_independently/Duke_all_data_cross-validation_with_fs_py2/'
	## DCIS-R
	best_c_train_on_all = np.load(duke_train_on_all_path + 'CV_ONLY_Training_NO_FS_400_LogisticRegression_c.npy', allow_pickle = True)
	uni_c_radiomics_nofs,counts_c_radiomics_nofs = np.unique(best_c_train_on_all,return_counts = True)
	most_selected_c_radiomics_nofs = uni_c_radiomics_nofs[np.flipud(np.argsort(counts_c_radiomics_nofs))][0]
	lr_radiomics_nofs = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 10000,C = most_selected_c_radiomics_nofs)
	_=lr_radiomics_nofs.fit(X_duke_normed, Y_duke)
	## DCIS-Rs
	idx_feat_picked_train_on_all = np.load(duke_train_on_all_path+'CV_ONLY_Training_WITH_FS_400_LogisticRegression_picked_features_idx.npy', allow_pickle = True).tolist()
	uni_radiomics_feats, counts_radiomics_feats = np.unique([item for sublist in idx_feat_picked_train_on_all for item in sublist], return_counts = True)
	selected_feature_idx = uni_radiomics_feats[np.flipud(np.argsort(counts_radiomics_feats))][:selected_radiomics_features_num]
	X_duke_normed_subset = np.copy(X_duke_normed[:, selected_feature_idx])
	selected_feature_names = np.array(radiomics_feat_list)[selected_feature_idx]
	#####
	best_c_train_duke_all_top_features = np.load(duke_train_on_all_path + 'With_%dTOP_features_from_CV_feature_selection_best_parameters_C.npy'%(selected_radiomics_features_num))
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
	[X_philips_optimam_normed, _] = mean_std_normalize(X_philips_optimam, [])
	[X_siemens_optimam_normed, _] = mean_std_normalize(X_siemens_optimam, [])
	X_optimam_normed = np.concatenate((X_hologic_optimam_normed, X_ge_optimam_normed, X_philips_optimam_normed, X_siemens_optimam_normed), axis= 0)
	Y_optimam = np.concatenate((Y_hologic_optimam, Y_ge_optimam, Y_philips_optimam, Y_siemens_optimam), axis= 0)
	info_optimam = info_optimam_hologic + info_optimam_ge + info_optimam_philips + info_optimam_siemens
	##
	proba_optimam_nofs = lr_radiomics_nofs.predict_proba(X_optimam_normed)[:,1]
	auc_optimam_nofs = metrics.roc_auc_score(Y_optimam,proba_optimam_nofs)
	auc_optimam_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_optimam, lr_radiomics_nofs.predict_proba(X_hologic_optimam_normed)[:,1])
	## Model A + Top Feats on CV 
	X_optimam_normed_subset = np.copy(X_optimam_normed[:, selected_feature_idx])
	proba_optimam_topfeats = lr_radiomics_topfeats.predict_proba(X_optimam_normed_subset)[:,1]
	auc_optimam_topfeats = metrics.roc_auc_score(Y_optimam,proba_optimam_topfeats)
	##
	auc_optimam_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_optimam, lr_radiomics_nofs.predict_proba(X_hologic_optimam_normed)[:,1])
	auc_optimam_topfeats_hologic_only = metrics.roc_auc_score(Y_hologic_optimam, lr_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1])
	auc_optimam_nofs_ge_only = metrics.roc_auc_score(Y_ge_optimam, lr_radiomics_nofs.predict_proba(X_ge_optimam_normed)[:,1])
	auc_optimam_topfeats_ge_only = metrics.roc_auc_score(Y_ge_optimam, lr_radiomics_topfeats.predict_proba(X_ge_optimam_normed[:, selected_feature_idx])[:,1])
	auc_optimam_nofs_philips_only = metrics.roc_auc_score(Y_philips_optimam, lr_radiomics_nofs.predict_proba(X_philips_optimam_normed)[:,1])
	auc_optimam_topfeats_philips_only = metrics.roc_auc_score(Y_philips_optimam, lr_radiomics_topfeats.predict_proba(X_philips_optimam_normed[:, selected_feature_idx])[:,1])
	auc_optimam_nofs_siemens_only = metrics.roc_auc_score(Y_siemens_optimam, lr_radiomics_nofs.predict_proba(X_siemens_optimam_normed)[:,1])
	auc_optimam_topfeats_siemens_only = metrics.roc_auc_score(Y_siemens_optimam, lr_radiomics_topfeats.predict_proba(X_siemens_optimam_normed[:, selected_feature_idx])[:,1])
	##
	print('OPTIMAM NO Feature Selection round(AUC: %.3f (GE: %.3f + Hologic: %.3f + Philips: %.3f + SIEMENS: %.3f); With TOP Feats: %.3f (GE: %.3f + Hologic: %.3f + Philips: %.3f + SIEMENS: %.3f)' 
		%(round(auc_optimam_nofs, 3), round(auc_optimam_nofs_ge_only, 3), round(auc_optimam_nofs_hologic_only, 3), round(auc_optimam_nofs_philips_only, 3), \
			round(auc_optimam_nofs_siemens_only, 3), round(auc_optimam_topfeats, 3), round(auc_optimam_topfeats_ge_only, 3), \
			round(auc_optimam_topfeats_hologic_only, 3), round(auc_optimam_topfeats_philips_only, 3), round(auc_optimam_topfeats_siemens_only, 3) ) )
	###
	with open(duke_train_on_all_path + 'Duke-all_models_test_on_optimam_hologic.csv','w') as f_csv_optimam_hologic:
		wri_csv_optimam_hologic = csv.writer(f_csv_optimam_hologic)
		_= wri_csv_optimam_hologic.writerow(['ID','labels','pred.nofs','pred.top.feats'])
		_=[wri_csv_optimam_hologic.writerow([info_optimam_hologic[idss][0], Y_hologic_optimam[idss], \
			lr_radiomics_nofs.predict_proba(X_hologic_optimam_normed)[idss,1], \
			lr_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[idss,1]]) for idss in range(Y_hologic_optimam.shape[0])]

	## Case AUC:
	optimam_unique_cases = np.unique([one_list[0] for one_list in info_optimam])
	Y_optimam_by_case = []
	proba_optimam_nofs_by_case_mean = []
	proba_optimam_nofs_by_case_max = []
	proba_optimam_topfeats_by_case_mean = []
	proba_optimam_topfeats_by_case_max = []
	for one_case in optimam_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam) if one_list[0]==one_case]
		proba_optimam_nofs_by_case_mean.append(np.mean([proba_optimam_nofs[ids] for ids in one_case_ids]))
		proba_optimam_nofs_by_case_max.append(np.max([proba_optimam_nofs[ids] for ids in one_case_ids]))
		proba_optimam_topfeats_by_case_mean.append(np.mean([proba_optimam_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_topfeats_by_case_max.append(np.max([proba_optimam_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_by_case.append([info_optimam[ids][2] for ids in one_case_ids][0])

	auc_optimam_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_nofs_by_case_mean))
	auc_optimam_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_nofs_by_case_max))
	auc_optimam_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_topfeats_by_case_mean))
	auc_optimam_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_topfeats_by_case_max))
	print('OPTIMAM-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_nofs_by_case_mean, 3), round(auc_optimam_nofs_by_case_max, 3), \
			round(auc_optimam_topfeats_by_case_mean, 3), round(auc_optimam_topfeats_by_case_max,3)))

	proba_optimam_hologic_nofs = lr_radiomics_nofs.predict_proba(X_hologic_optimam_normed)[:,1]
	proba_optimam_hologic_topfeats = lr_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1]
	##
	optimam_hologic_unique_cases = np.unique([one_list[0] for one_list in info_optimam_hologic])
	Y_optimam_hologic_by_case = []
	proba_optimam_hologic_nofs_by_case_mean = []
	proba_optimam_hologic_nofs_by_case_max = []
	proba_optimam_hologic_topfeats_by_case_mean = []
	proba_optimam_hologic_topfeats_by_case_max = []
	for one_case in optimam_hologic_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_hologic) if one_list[0]==one_case]
		proba_optimam_hologic_nofs_by_case_mean.append(np.mean([proba_optimam_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_hologic_nofs_by_case_max.append(np.max([proba_optimam_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_hologic_topfeats_by_case_mean.append(np.mean([proba_optimam_hologic_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_hologic_topfeats_by_case_max.append(np.max([proba_optimam_hologic_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_hologic_by_case.append([info_optimam_hologic[ids][2] for ids in one_case_ids][0])

	auc_optimam_hologic_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_nofs_by_case_mean))
	auc_optimam_hologic_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_nofs_by_case_max))
	auc_optimam_hologic_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_topfeats_by_case_mean))
	auc_optimam_hologic_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_topfeats_by_case_max))
	print('OPTIMAM-Hologic Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_hologic_nofs_by_case_mean, 3), round(auc_optimam_hologic_nofs_by_case_max, 3), \
			round(auc_optimam_hologic_topfeats_by_case_mean, 3), round(auc_optimam_hologic_topfeats_by_case_max,3)))


	##################################################################
	##################################################################
	#### AUC Performance On NKI
	[X_hologic_nki_normed,_] = mean_std_normalize(X_hologic_nki, [])
	[X_ge_nki_normed,_] = mean_std_normalize(X_ge_nki, [])
	[X_siemens_nki_normed,_] = mean_std_normalize(X_siemens_nki, [])
	X_nki_normed = np.concatenate((X_hologic_nki_normed, X_ge_nki_normed, X_siemens_nki_normed),axis= 0)
	Y_nki = np.concatenate((Y_hologic_nki, Y_ge_nki, Y_siemens_nki),axis= 0)
	info_nki = info_nki_hologic + info_nki_ge + info_nki_siemens
	proba_nki_nofs_hologic_only = lr_radiomics_nofs.predict_proba(X_hologic_nki_normed)[:,1]
	auc_nki_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_nki, proba_nki_nofs_hologic_only)
	####
	proba_nki_nofs = lr_radiomics_nofs.predict_proba(X_nki_normed)[:,1]
	auc_nki_nofs = metrics.roc_auc_score(Y_nki,proba_nki_nofs)
	X_nki_normed_subset = np.copy(X_nki_normed[:, selected_feature_idx])
	proba_nki_topfeats = lr_radiomics_topfeats.predict_proba(X_nki_normed_subset)[:,1]
	auc_nki_topfeats = metrics.roc_auc_score(Y_nki,proba_nki_topfeats)
	##
	auc_nki_nofs_ge_only = metrics.roc_auc_score(Y_ge_nki, lr_radiomics_nofs.predict_proba(X_ge_nki_normed)[:,1])
	auc_nki_topfeats_ge_only = metrics.roc_auc_score(Y_ge_nki, lr_radiomics_topfeats.predict_proba(X_ge_nki_normed[:, selected_feature_idx])[:,1])
	auc_nki_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_nki, lr_radiomics_nofs.predict_proba(X_hologic_nki_normed)[:,1])
	auc_nki_topfeats_hologic_only = metrics.roc_auc_score(Y_hologic_nki, lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1])
	# auc_nki_nofs_siemens_only = metrics.roc_auc_score(Y_siemens_nki, lr_radiomics_nofs.predict_proba(X_siemens_nki_normed)[:,1])
	# auc_nki_topfeats_siemens_only = metrics.roc_auc_score(Y_siemens_nki, lr_radiomics_topfeats.predict_proba(X_siemens_nki_normed[:, selected_feature_idx])[:,1])
	## No Upstage cases for SIEMENS
	##
	print('NKI NO Feature Selection round(AUC: %.3f (GE: %.3f + Hologic: %.3f); With TOP Feats: %.3f (GE: %.3f + Hologic: %.3f)' 
		%(round(auc_nki_nofs, 3), round(auc_nki_nofs_ge_only, 3), round(auc_nki_nofs_hologic_only, 3), \
			round(auc_nki_topfeats, 3), round(auc_nki_topfeats_ge_only, 3), round(auc_nki_topfeats_hologic_only, 3)) )

	proba_nki_hologic_nofs = lr_radiomics_nofs.predict_proba(X_hologic_nki_normed)[:,1]
	proba_nki_hologic_topfeats = lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1]
	##
	nki_hologic_unique_cases = np.unique([one_list[0] for one_list in info_nki_hologic])
	Y_nki_hologic_by_case = []
	proba_nki_hologic_nofs_by_case_mean = []
	proba_nki_hologic_nofs_by_case_max = []
	proba_nki_hologic_topfeats_by_case_mean = []
	proba_nki_hologic_topfeats_by_case_max = []
	for one_case in nki_hologic_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_nki_hologic) if one_list[0]==one_case]
		proba_nki_hologic_nofs_by_case_mean.append(np.mean([proba_nki_hologic_nofs[ids] for ids in one_case_ids]))
		proba_nki_hologic_nofs_by_case_max.append(np.max([proba_nki_hologic_nofs[ids] for ids in one_case_ids]))
		proba_nki_hologic_topfeats_by_case_mean.append(np.mean([proba_nki_hologic_topfeats[ids] for ids in one_case_ids]))
		proba_nki_hologic_topfeats_by_case_max.append(np.max([proba_nki_hologic_topfeats[ids] for ids in one_case_ids]))
		Y_nki_hologic_by_case.append([info_nki_hologic[ids][2] for ids in one_case_ids][0])

	auc_nki_hologic_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_nofs_by_case_mean))
	auc_nki_hologic_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_nofs_by_case_max))
	auc_nki_hologic_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_topfeats_by_case_mean))
	auc_nki_hologic_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_topfeats_by_case_max))
	print('NKI Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_nki_hologic_nofs_by_case_mean, 3), round(auc_nki_hologic_nofs_by_case_max, 3), \
			round(auc_nki_hologic_topfeats_by_case_mean, 3), round(auc_nki_hologic_topfeats_by_case_max,3)))
	

	with open(duke_train_on_all_path + 'Duke-all_models_test_on_nki_hologic.csv','w') as f_csv_nki_hologic:
		wri_csv_nki_hologic = csv.writer(f_csv_nki_hologic)
		_= wri_csv_nki_hologic.writerow(['ID','labels','pred.nofs','pred.top.feats'])
		_=[wri_csv_nki_hologic.writerow([info_nki_hologic[idss][0], Y_hologic_nki[idss], lr_radiomics_nofs.predict_proba(X_hologic_nki_normed)[idss,1], \
			lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[idss,1]]) for idss in range(Y_hologic_nki.shape[0])]

	# OPTIMAM NO Feature Selection round(AUC: 0.604 (GE: 0.530 + Hologic: 0.599 + Philips: 0.646 + SIEMENS: 0.619); With TOP Feats: 0.606 (GE: 0.494 + Hologic: 0.604 + Philips: 0.646 + SIEMENS: 0.647)
	# OPTIMAM-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: 0.610, Case MAX: 0.611; WITH Top Feats Case MEAN: 0.614, Case MAX: 0.612
	# OPTIMAM-Hologic Testing AUCs (Model From NKI-Hologic), No FS Case MEAN: 0.606, Case MAX: 0.608; WITH Top Feats Case MEAN: 0.613, Case MAX: 0.614
	# NKI NO Feature Selection round(AUC: 0.651 (GE: 0.538 + Hologic: 0.658); With TOP Feats: 0.657 (GE: 0.538 + Hologic: 0.663)
	# NKI Testing AUCs (Model From DUKE), No FS Case MEAN: 0.664, Case MAX: 0.692; WITH Top Feats Case MEAN: 0.668, Case MAX: 0.693

	##############################
	##############################
	
	## OPTIMAM Magviews
	magviews_case_root_dir = 'OPTIMAM/magview_features_temp/'
	optimam_magviews_xlsx_file_name = magviews_case_root_dir + 'Rui_DCIS_Cases_Diagnosis_Merged_Information.xlsx'
	optimam_magviews_file = openpyxl.load_workbook(optimam_magviews_xlsx_file_name)
	optimam_magviews_image_sheet = optimam_magviews_file.worksheets[1]
	optimam_magviews_image_item_list = [str(optimam_magviews_image_sheet.cell(row = 1, column = col_now).value) for col_now in range(1, optimam_magviews_image_sheet.max_column + 1)]
	optimam_magviews_caseid_idx = [col_now + 1 for col_now, ii in enumerate(optimam_magviews_image_item_list) if 'ClientID' in ii][0]
	optimam_magviews_imageid_idx = [col_now + 1 for col_now, ii in enumerate(optimam_magviews_image_item_list) if 'ImageSOPIUID' in ii][0]
	optimam_magviews_diagnosis_idx = [col_now + 1 for col_now, ii in enumerate(optimam_magviews_image_item_list) if 'Diagnosis' in ii][0]
	optimam_magviews_model_idx = [col_now + 1 for col_now, ii in enumerate(optimam_magviews_image_item_list) if 'ManufacturersModelName' in ii][0]
	optimam_magviews_caseid_image_diagnosis_model = [[str(optimam_magviews_image_sheet.cell(row = idds, column = optimam_magviews_caseid_idx).value), \
			str(optimam_magviews_image_sheet.cell(row = idds, column = optimam_magviews_imageid_idx).value), \
			str(optimam_magviews_image_sheet.cell(row = idds, column = optimam_magviews_diagnosis_idx).value), \
			str(optimam_magviews_image_sheet.cell(row = idds, column = optimam_magviews_model_idx).value)] for idds in range(2, optimam_magviews_image_sheet.max_row+1)]

	optimam_magviews_features_all = glob.glob(magviews_case_root_dir + 'GE/Feature109_Final_with_unet_ver11-ep260/*.npy') + glob.glob(magviews_case_root_dir + 'Non-GE/Feature109_Final_with_unet_ver11-ep260/*.npy')
	optimam_magviews_caseid_image_diagnosis_model_with_feats = []
	for one_feat in optimam_magviews_features_all:
		optimam_magviews_caseid_image_diagnosis_model_with_feats.append([ss + [one_feat] for ss in optimam_magviews_caseid_image_diagnosis_model if ss[1] in one_feat][0])

	##
	X_ge_optimam_magviews = np.array([]).reshape(0,num_radiomic_features)
	Y_ge_optimam_magviews = np.array([]).astype(np.uint8)
	X_hologic_optimam_magviews = np.array([]).reshape(0,num_radiomic_features)
	Y_hologic_optimam_magviews = np.array([]).astype(np.uint8)
	info_optimam_magviews_hologic = []
	info_optimam_magviews_ge = []

	for one_id, one_image, one_dx, one_model, one_added_feat in optimam_magviews_caseid_image_diagnosis_model_with_feats:
		X_one_feat = np.load(one_added_feat).reshape((1,num_radiomic_features))
		if 'Selenia' in one_model:
			X_hologic_optimam_magviews = np.concatenate((X_hologic_optimam_magviews,X_one_feat),axis = 0)
			Y_hologic_optimam_magviews = np.concatenate((Y_hologic_optimam_magviews,[0 if 'pure' in one_dx.lower() else 1]))
			info_optimam_magviews_hologic.append([one_id, one_image, 0 if 'pure' in one_dx.lower() else 1, one_model, one_added_feat])
		if 'Senograph' in one_model:
			X_ge_optimam_magviews = np.concatenate((X_ge_optimam_magviews,X_one_feat),axis = 0)
			Y_ge_optimam_magviews = np.concatenate((Y_ge_optimam_magviews,[0 if 'pure' in one_dx.lower() else 1]))
			info_optimam_magviews_ge.append([one_id, one_image, 0 if 'pure' in one_dx.lower() else 1, one_model, one_added_feat])
	
	[X_ge_optimam_magviews_normed, _] = mean_std_normalize(X_ge_optimam_magviews, [])
	[X_hologic_optimam_magviews_normed, _] = mean_std_normalize(X_hologic_optimam_magviews, [])
	X_optimam_magviews_normed = np.concatenate((X_ge_optimam_magviews_normed,X_hologic_optimam_magviews_normed),axis= 0)
	Y_optimam_magviews = np.concatenate((Y_ge_optimam_magviews, Y_hologic_optimam_magviews),axis= 0)
	info_optimam_magviews = info_optimam_magviews_ge + info_optimam_magviews_hologic
	##
	proba_optimam_magviews_nofs = lr_radiomics_nofs.predict_proba(X_optimam_magviews_normed)[:,1]
	proba_optimam_magviews_topfeats = lr_radiomics_topfeats.predict_proba(X_optimam_magviews_normed[:, selected_feature_idx])[:,1]
	##
	auc_optimam_magviews_nofs = metrics.roc_auc_score(Y_optimam_magviews,proba_optimam_magviews_nofs)
	auc_optimam_magviews_topfeats = metrics.roc_auc_score(Y_optimam_magviews,proba_optimam_magviews_topfeats)
	auc_optimam_magviews_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_optimam_magviews, lr_radiomics_nofs.predict_proba(X_hologic_optimam_magviews_normed)[:,1])
	auc_optimam_magviews_topfeats_hologic_only = metrics.roc_auc_score(Y_hologic_optimam_magviews, lr_radiomics_topfeats.predict_proba(X_hologic_optimam_magviews_normed[:, selected_feature_idx])[:,1])
	auc_optimam_magviews_nofs_ge_only = metrics.roc_auc_score(Y_ge_optimam_magviews, lr_radiomics_nofs.predict_proba(X_ge_optimam_magviews_normed)[:,1])
	auc_optimam_magviews_topfeats_ge_only = metrics.roc_auc_score(Y_ge_optimam_magviews, lr_radiomics_topfeats.predict_proba(X_ge_optimam_magviews_normed[:, selected_feature_idx])[:,1])
	print('OPTIMAM_MAGVIEWS NO Feature Selection round(AUC: %.3f (GE: %.3f + Hologic: %.3f); With TOP Feats: %.3f (GE: %.3f + Hologic: %.3f)' 
		%(round(auc_optimam_magviews_nofs, 3), round(auc_optimam_magviews_nofs_ge_only, 3), round(auc_optimam_magviews_nofs_hologic_only, 3), \
			round(auc_optimam_magviews_topfeats, 3), round(auc_optimam_magviews_topfeats_ge_only, 3), round(auc_optimam_magviews_topfeats_hologic_only, 3) ))
	##############################
	##############################
	## Case ALL AUC:
	optimam_magviews_unique_cases = np.unique([one_list[0] for one_list in info_optimam_magviews])
	Y_optimam_magviews_by_case = []
	proba_optimam_magviews_nofs_by_case_mean = []
	proba_optimam_magviews_nofs_by_case_max = []
	proba_optimam_magviews_topfeats_by_case_mean = []
	proba_optimam_magviews_topfeats_by_case_max = []
	for one_case in optimam_magviews_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_magviews) if one_list[0]==one_case]
		proba_optimam_magviews_nofs_by_case_mean.append(np.mean([proba_optimam_magviews_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_nofs_by_case_max.append(np.max([proba_optimam_magviews_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_topfeats_by_case_mean.append(np.mean([proba_optimam_magviews_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_magviews_topfeats_by_case_max.append(np.max([proba_optimam_magviews_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_magviews_by_case.append([info_optimam_magviews[ids][2] for ids in one_case_ids][0])

	auc_optimam_magviews_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_nofs_by_case_mean))
	auc_optimam_magviews_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_nofs_by_case_max))
	auc_optimam_magviews_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_topfeats_by_case_mean))
	auc_optimam_magviews_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_topfeats_by_case_max))
	print('OPTIMAM_MAGVIEWS-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_magviews_nofs_by_case_mean, 3), round(auc_optimam_magviews_nofs_by_case_max, 3), \
			round(auc_optimam_magviews_topfeats_by_case_mean, 3), round(auc_optimam_magviews_topfeats_by_case_max,3)))
	##############################
	##############################
	## OPTIMAM Hologic
	proba_optimam_magviews_hologic_nofs = lr_radiomics_nofs.predict_proba(X_hologic_optimam_magviews_normed)[:,1]
	proba_optimam_magviews_hologic_topfeats = lr_radiomics_topfeats.predict_proba(X_hologic_optimam_magviews_normed[:, selected_feature_idx])[:,1]
	##
	optimam_magviews_hologic_unique_cases = np.unique([one_list[0] for one_list in info_optimam_magviews_hologic])
	Y_optimam_magviews_hologic_by_case = []
	proba_optimam_magviews_hologic_nofs_by_case_mean = []
	proba_optimam_magviews_hologic_nofs_by_case_max = []
	proba_optimam_magviews_hologic_topfeats_by_case_mean = []
	proba_optimam_magviews_hologic_topfeats_by_case_max = []
	for one_case in optimam_magviews_hologic_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_magviews_hologic) if one_list[0]==one_case]
		proba_optimam_magviews_hologic_nofs_by_case_mean.append(np.mean([proba_optimam_magviews_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_hologic_nofs_by_case_max.append(np.max([proba_optimam_magviews_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_hologic_topfeats_by_case_mean.append(np.mean([proba_optimam_magviews_hologic_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_magviews_hologic_topfeats_by_case_max.append(np.max([proba_optimam_magviews_hologic_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_magviews_hologic_by_case.append([info_optimam_magviews_hologic[ids][2] for ids in one_case_ids][0])

	auc_optimam_magviews_hologic_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_nofs_by_case_mean))
	auc_optimam_magviews_hologic_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_nofs_by_case_max))
	auc_optimam_magviews_hologic_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_topfeats_by_case_mean))
	auc_optimam_magviews_hologic_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_topfeats_by_case_max))
	print('OPTIMAM_MAGVIEWS-Hologic Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_magviews_hologic_nofs_by_case_mean, 3), round(auc_optimam_magviews_hologic_nofs_by_case_max, 3), \
			round(auc_optimam_magviews_hologic_topfeats_by_case_mean, 3), round(auc_optimam_magviews_hologic_topfeats_by_case_max,3)))
	##############################
	##############################
	## OPTIMAM GE
	proba_optimam_magviews_ge_nofs = lr_radiomics_nofs.predict_proba(X_ge_optimam_magviews_normed)[:,1]
	proba_optimam_magviews_ge_topfeats = lr_radiomics_topfeats.predict_proba(X_ge_optimam_magviews_normed[:, selected_feature_idx])[:,1]
	##
	optimam_magviews_ge_unique_cases = np.unique([one_list[0] for one_list in info_optimam_magviews_ge])
	Y_optimam_magviews_ge_by_case = []
	proba_optimam_magviews_ge_nofs_by_case_mean = []
	proba_optimam_magviews_ge_nofs_by_case_max = []
	proba_optimam_magviews_ge_topfeats_by_case_mean = []
	proba_optimam_magviews_ge_topfeats_by_case_max = []
	for one_case in optimam_magviews_ge_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_magviews_ge) if one_list[0]==one_case]
		proba_optimam_magviews_ge_nofs_by_case_mean.append(np.mean([proba_optimam_magviews_ge_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_ge_nofs_by_case_max.append(np.max([proba_optimam_magviews_ge_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_ge_topfeats_by_case_mean.append(np.mean([proba_optimam_magviews_ge_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_magviews_ge_topfeats_by_case_max.append(np.max([proba_optimam_magviews_ge_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_magviews_ge_by_case.append([info_optimam_magviews_ge[ids][2] for ids in one_case_ids][0])

	auc_optimam_magviews_ge_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_nofs_by_case_mean))
	auc_optimam_magviews_ge_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_nofs_by_case_max))
	auc_optimam_magviews_ge_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_topfeats_by_case_mean))
	auc_optimam_magviews_ge_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_topfeats_by_case_max))
	print('OPTIMAM_MAGVIEWS-GE Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_magviews_ge_nofs_by_case_mean, 3), round(auc_optimam_magviews_ge_nofs_by_case_max, 3), \
			round(auc_optimam_magviews_ge_topfeats_by_case_mean, 3), round(auc_optimam_magviews_ge_topfeats_by_case_max,3)))

	
	# OPTIMAM_MAGVIEWS NO Feature Selection round(AUC: 0.604 (GE: 0.715 + Hologic: 0.544); With TOP Feats: 0.615 (GE: 0.727 + Hologic: 0.552)
	# OPTIMAM_MAGVIEWS-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: 0.613, Case MAX: 0.617; WITH Top Feats Case MEAN: 0.623, Case MAX: 0.638
	# OPTIMAM_MAGVIEWS-Hologic Testing AUCs (Model From DUKE), No FS Case MEAN: 0.549, Case MAX: 0.553; WITH Top Feats Case MEAN: 0.552, Case MAX: 0.567
	# OPTIMAM_MAGVIEWS-GE Testing AUCs (Model From DUKE), No FS Case MEAN: 0.723, Case MAX: 0.730; WITH Top Feats Case MEAN: 0.740, Case MAX: 0.749
	
	info_duke_ge
	##############################################################################
	selected_radiomics_features_num = 11
	duke_train_on_ge_path = 'training_results/results_train_ge_hologic_separately/Train_only_ge/'
	## DCIS-R
	best_c_train_on_ge = np.load(duke_train_on_ge_path + 'CV_ONLY_Training_NO_FS_400_LogisticRegression_c.npy', allow_pickle = True)
	uni_c_radiomics_nofs,counts_c_radiomics_nofs = np.unique(best_c_train_on_ge,return_counts = True)
	most_selected_c_radiomics_nofs = uni_c_radiomics_nofs[np.flipud(np.argsort(counts_c_radiomics_nofs))][0]
	lr_duke_ge_radiomics_nofs = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 10000,C = most_selected_c_radiomics_nofs)
	_=lr_duke_ge_radiomics_nofs.fit(X_ge_duke_normed, Y_ge_duke)
	## DCIS-Rs
	idx_feat_picked_train_on_ge = np.load(duke_train_on_ge_path+'CV_ONLY_Training_WITH_FS_400_LogisticRegression_picked_features_idx.npy', allow_pickle = True).tolist()
	uni_radiomics_feats, counts_radiomics_feats = np.unique([item for sublist in idx_feat_picked_train_on_ge for item in sublist], return_counts = True)
	selected_feature_idx = uni_radiomics_feats[np.flipud(np.argsort(counts_radiomics_feats))][:selected_radiomics_features_num]
	X_ge_duke_normed_subset = np.copy(X_ge_duke_normed[:, selected_feature_idx])
	selected_feature_names = np.array(radiomics_feat_list)[selected_feature_idx]
	#####
	best_c_train_on_ge_top_features = np.load(duke_train_on_ge_path + 'With_%dTOP_features_from_CV_feature_selection_best_parameters_C.npy'%(selected_radiomics_features_num))
	uni_c_train_on_ge_topfeats, counts_c_train_on_ge_topfeats = np.unique(best_c_train_on_ge_top_features,return_counts = True)
	most_selected_c_train_on_ge_top_feats = uni_c_train_on_ge_topfeats[np.flipud(np.argsort(counts_c_train_on_ge_topfeats))][0]
	lr_duke_ge_radiomics_topfeats = linear_model.LogisticRegression(penalty='l2',solver='lbfgs',max_iter = 10000,C = most_selected_c_train_on_ge_top_feats)
	_=lr_duke_ge_radiomics_topfeats.fit(X_ge_duke_normed_subset,Y_ge_duke)
	##
	##################################################################
	##################################################################
	#### AUC Performance On OPTIMAM
	[X_hologic_optimam_normed, _] = mean_std_normalize(X_hologic_optimam, [])
	[X_ge_optimam_normed, _] = mean_std_normalize(X_ge_optimam, [])
	[X_philips_optimam_normed, _] = mean_std_normalize(X_philips_optimam, [])
	[X_siemens_optimam_normed, _] = mean_std_normalize(X_siemens_optimam, [])
	X_optimam_normed = np.concatenate((X_hologic_optimam_normed, X_ge_optimam_normed, X_philips_optimam_normed, X_siemens_optimam_normed), axis= 0)
	Y_optimam = np.concatenate((Y_hologic_optimam, Y_ge_optimam, Y_philips_optimam, Y_siemens_optimam), axis= 0)
	info_optimam = info_optimam_hologic + info_optimam_ge + info_optimam_philips + info_optimam_siemens
	##
	proba_optimam_nofs = lr_duke_ge_radiomics_nofs.predict_proba(X_optimam_normed)[:,1]
	auc_optimam_nofs = metrics.roc_auc_score(Y_optimam,proba_optimam_nofs)
	auc_optimam_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_optimam, lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_optimam_normed)[:,1])
	## Model A + Top Feats on CV 
	X_optimam_normed_subset = np.copy(X_optimam_normed[:, selected_feature_idx])
	proba_optimam_topfeats = lr_duke_ge_radiomics_topfeats.predict_proba(X_optimam_normed_subset)[:,1]
	auc_optimam_topfeats = metrics.roc_auc_score(Y_optimam,proba_optimam_topfeats)
	##
	auc_optimam_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_optimam, lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_optimam_normed)[:,1])
	auc_optimam_topfeats_hologic_only = metrics.roc_auc_score(Y_hologic_optimam, lr_duke_ge_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1])
	auc_optimam_nofs_ge_only = metrics.roc_auc_score(Y_ge_optimam, lr_duke_ge_radiomics_nofs.predict_proba(X_ge_optimam_normed)[:,1])
	auc_optimam_topfeats_ge_only = metrics.roc_auc_score(Y_ge_optimam, lr_duke_ge_radiomics_topfeats.predict_proba(X_ge_optimam_normed[:, selected_feature_idx])[:,1])
	auc_optimam_nofs_philips_only = metrics.roc_auc_score(Y_philips_optimam, lr_duke_ge_radiomics_nofs.predict_proba(X_philips_optimam_normed)[:,1])
	auc_optimam_topfeats_philips_only = metrics.roc_auc_score(Y_philips_optimam, lr_duke_ge_radiomics_topfeats.predict_proba(X_philips_optimam_normed[:, selected_feature_idx])[:,1])
	auc_optimam_nofs_siemens_only = metrics.roc_auc_score(Y_siemens_optimam, lr_duke_ge_radiomics_nofs.predict_proba(X_siemens_optimam_normed)[:,1])
	auc_optimam_topfeats_siemens_only = metrics.roc_auc_score(Y_siemens_optimam, lr_duke_ge_radiomics_topfeats.predict_proba(X_siemens_optimam_normed[:, selected_feature_idx])[:,1])

	##
	print('OPTIMAM NO Feature Selection round(AUC: %.3f (GE: %.3f + Hologic: %.3f + Philips: %.3f + SIEMENS: %.3f); With TOP Feats: %.3f (GE: %.3f + Hologic: %.3f + Philips: %.3f + SIEMENS: %.3f)' 
		%(round(auc_optimam_nofs, 3), round(auc_optimam_nofs_ge_only, 3), round(auc_optimam_nofs_hologic_only, 3), round(auc_optimam_nofs_philips_only, 3), \
			round(auc_optimam_nofs_siemens_only, 3), round(auc_optimam_topfeats, 3), round(auc_optimam_topfeats_ge_only, 3), \
			round(auc_optimam_topfeats_hologic_only, 3), round(auc_optimam_topfeats_philips_only, 3), round(auc_optimam_topfeats_siemens_only, 3) ) )
	## Case AUC:
	optimam_unique_cases = np.unique([one_list[0] for one_list in info_optimam])
	Y_optimam_by_case = []
	proba_optimam_nofs_by_case_mean = []
	proba_optimam_nofs_by_case_max = []
	proba_optimam_topfeats_by_case_mean = []
	proba_optimam_topfeats_by_case_max = []
	for one_case in optimam_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam) if one_list[0]==one_case]
		proba_optimam_nofs_by_case_mean.append(np.mean([proba_optimam_nofs[ids] for ids in one_case_ids]))
		proba_optimam_nofs_by_case_max.append(np.max([proba_optimam_nofs[ids] for ids in one_case_ids]))
		proba_optimam_topfeats_by_case_mean.append(np.mean([proba_optimam_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_topfeats_by_case_max.append(np.max([proba_optimam_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_by_case.append([info_optimam[ids][2] for ids in one_case_ids][0])

	auc_optimam_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_nofs_by_case_mean))
	auc_optimam_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_nofs_by_case_max))
	auc_optimam_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_topfeats_by_case_mean))
	auc_optimam_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_by_case), np.array(proba_optimam_topfeats_by_case_max))
	print('OPTIMAM-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_nofs_by_case_mean, 3), round(auc_optimam_nofs_by_case_max, 3), \
			round(auc_optimam_topfeats_by_case_mean, 3), round(auc_optimam_topfeats_by_case_max,3)))

	proba_optimam_hologic_nofs = lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_optimam_normed)[:,1]
	proba_optimam_hologic_topfeats = lr_duke_ge_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1]
	##
	optimam_hologic_unique_cases = np.unique([one_list[0] for one_list in info_optimam_hologic])
	Y_optimam_hologic_by_case = []
	proba_optimam_hologic_nofs_by_case_mean = []
	proba_optimam_hologic_nofs_by_case_max = []
	proba_optimam_hologic_topfeats_by_case_mean = []
	proba_optimam_hologic_topfeats_by_case_max = []
	for one_case in optimam_hologic_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_hologic) if one_list[0]==one_case]
		proba_optimam_hologic_nofs_by_case_mean.append(np.mean([proba_optimam_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_hologic_nofs_by_case_max.append(np.max([proba_optimam_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_hologic_topfeats_by_case_mean.append(np.mean([proba_optimam_hologic_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_hologic_topfeats_by_case_max.append(np.max([proba_optimam_hologic_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_hologic_by_case.append([info_optimam_hologic[ids][2] for ids in one_case_ids][0])

	auc_optimam_hologic_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_nofs_by_case_mean))
	auc_optimam_hologic_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_nofs_by_case_max))
	auc_optimam_hologic_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_topfeats_by_case_mean))
	auc_optimam_hologic_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_hologic_by_case), np.array(proba_optimam_hologic_topfeats_by_case_max))
	print('OPTIMAM-Hologic Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_hologic_nofs_by_case_mean, 3), round(auc_optimam_hologic_nofs_by_case_max, 3), \
			round(auc_optimam_hologic_topfeats_by_case_mean, 3), round(auc_optimam_hologic_topfeats_by_case_max,3)))


	##################################################################
	##################################################################
	#### AUC Performance On NKI
	[X_hologic_nki_normed,_] = mean_std_normalize(X_hologic_nki, [])
	[X_ge_nki_normed,_] = mean_std_normalize(X_ge_nki, [])
	[X_siemens_nki_normed,_] = mean_std_normalize(X_siemens_nki, [])
	X_nki_normed = np.concatenate((X_hologic_nki_normed, X_ge_nki_normed, X_siemens_nki_normed),axis= 0)
	Y_nki = np.concatenate((Y_hologic_nki, Y_ge_nki, Y_siemens_nki),axis= 0)
	info_nki = info_nki_hologic + info_nki_ge + info_nki_siemens
	proba_nki_nofs_hologic_only = lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_nki_normed)[:,1]
	auc_nki_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_nki, proba_nki_nofs_hologic_only)
	####
	proba_nki_nofs = lr_duke_ge_radiomics_nofs.predict_proba(X_nki_normed)[:,1]
	auc_nki_nofs = metrics.roc_auc_score(Y_nki,proba_nki_nofs)
	X_nki_normed_subset = np.copy(X_nki_normed[:, selected_feature_idx])
	proba_nki_topfeats = lr_duke_ge_radiomics_topfeats.predict_proba(X_nki_normed_subset)[:,1]
	auc_nki_topfeats = metrics.roc_auc_score(Y_nki,proba_nki_topfeats)
	##
	auc_nki_nofs_ge_only = metrics.roc_auc_score(Y_ge_nki, lr_duke_ge_radiomics_nofs.predict_proba(X_ge_nki_normed)[:,1])
	auc_nki_topfeats_ge_only = metrics.roc_auc_score(Y_ge_nki, lr_duke_ge_radiomics_topfeats.predict_proba(X_ge_nki_normed[:, selected_feature_idx])[:,1])
	auc_nki_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_nki, lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_nki_normed)[:,1])
	auc_nki_topfeats_hologic_only = metrics.roc_auc_score(Y_hologic_nki, lr_duke_ge_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1])
	# auc_nki_nofs_siemens_only = metrics.roc_auc_score(Y_siemens_nki, lr_duke_ge_radiomics_nofs.predict_proba(X_siemens_nki_normed)[:,1])
	# auc_nki_topfeats_siemens_only = metrics.roc_auc_score(Y_siemens_nki, lr_duke_ge_radiomics_topfeats.predict_proba(X_siemens_nki_normed[:, selected_feature_idx])[:,1])
	## No Upstage cases for SIEMENS
	##
	print('NKI NO Feature Selection round(AUC: %.3f (GE: %.3f + Hologic: %.3f); With TOP Feats: %.3f (GE: %.3f + Hologic: %.3f)' 
		%(round(auc_nki_nofs, 3), round(auc_nki_nofs_ge_only, 3), round(auc_nki_nofs_hologic_only, 3), \
			round(auc_nki_topfeats, 3), round(auc_nki_topfeats_ge_only, 3), round(auc_nki_topfeats_hologic_only, 3)) )

	proba_nki_hologic_nofs = lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_nki_normed)[:,1]
	proba_nki_hologic_topfeats = lr_duke_ge_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1]
	##
	nki_hologic_unique_cases = np.unique([one_list[0] for one_list in info_nki_hologic])
	Y_nki_hologic_by_case = []
	proba_nki_hologic_nofs_by_case_mean = []
	proba_nki_hologic_nofs_by_case_max = []
	proba_nki_hologic_topfeats_by_case_mean = []
	proba_nki_hologic_topfeats_by_case_max = []
	for one_case in nki_hologic_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_nki_hologic) if one_list[0]==one_case]
		proba_nki_hologic_nofs_by_case_mean.append(np.mean([proba_nki_hologic_nofs[ids] for ids in one_case_ids]))
		proba_nki_hologic_nofs_by_case_max.append(np.max([proba_nki_hologic_nofs[ids] for ids in one_case_ids]))
		proba_nki_hologic_topfeats_by_case_mean.append(np.mean([proba_nki_hologic_topfeats[ids] for ids in one_case_ids]))
		proba_nki_hologic_topfeats_by_case_max.append(np.max([proba_nki_hologic_topfeats[ids] for ids in one_case_ids]))
		Y_nki_hologic_by_case.append([info_nki_hologic[ids][2] for ids in one_case_ids][0])

	auc_nki_hologic_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_nofs_by_case_mean))
	auc_nki_hologic_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_nofs_by_case_max))
	auc_nki_hologic_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_topfeats_by_case_mean))
	auc_nki_hologic_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_nki_hologic_by_case), np.array(proba_nki_hologic_topfeats_by_case_max))
	print('NKI Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_nki_hologic_nofs_by_case_mean, 3), round(auc_nki_hologic_nofs_by_case_max, 3), \
			round(auc_nki_hologic_topfeats_by_case_mean, 3), round(auc_nki_hologic_topfeats_by_case_max,3)))
	
	# OPTIMAM NO Feature Selection round(AUC: 0.604 (GE: 0.530 + Hologic: 0.599 + Philips: 0.646 + SIEMENS: 0.619); With TOP Feats: 0.606 (GE: 0.494 + Hologic: 0.604 + Philips: 0.646 + SIEMENS: 0.647)
	# OPTIMAM-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: 0.610, Case MAX: 0.611; WITH Top Feats Case MEAN: 0.614, Case MAX: 0.612
	# OPTIMAM-Hologic Testing AUCs (Model From NKI-Hologic), No FS Case MEAN: 0.606, Case MAX: 0.608; WITH Top Feats Case MEAN: 0.613, Case MAX: 0.614
	# NKI NO Feature Selection round(AUC: 0.651 (GE: 0.538 + Hologic: 0.658); With TOP Feats: 0.657 (GE: 0.538 + Hologic: 0.663)
	# NKI Testing AUCs (Model From DUKE), No FS Case MEAN: 0.664, Case MAX: 0.692; WITH Top Feats Case MEAN: 0.668, Case MAX: 0.693
	##############################
	##############################
	## OPTIMAM Magviews
	[X_ge_optimam_magviews_normed, _] = mean_std_normalize(X_ge_optimam_magviews, [])
	[X_hologic_optimam_magviews_normed, _] = mean_std_normalize(X_hologic_optimam_magviews, [])
	X_optimam_magviews_normed = np.concatenate((X_ge_optimam_magviews_normed,X_hologic_optimam_magviews_normed),axis= 0)
	Y_optimam_magviews = np.concatenate((Y_ge_optimam_magviews, Y_hologic_optimam_magviews),axis= 0)
	info_optimam_magviews = info_optimam_magviews_ge + info_optimam_magviews_hologic
	##
	proba_optimam_magviews_nofs = lr_duke_ge_radiomics_nofs.predict_proba(X_optimam_magviews_normed)[:,1]
	proba_optimam_magviews_topfeats = lr_duke_ge_radiomics_topfeats.predict_proba(X_optimam_magviews_normed[:, selected_feature_idx])[:,1]
	##
	auc_optimam_magviews_nofs = metrics.roc_auc_score(Y_optimam_magviews,proba_optimam_magviews_nofs)
	auc_optimam_magviews_topfeats = metrics.roc_auc_score(Y_optimam_magviews,proba_optimam_magviews_topfeats)
	auc_optimam_magviews_nofs_hologic_only = metrics.roc_auc_score(Y_hologic_optimam_magviews, lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_optimam_magviews_normed)[:,1])
	auc_optimam_magviews_topfeats_hologic_only = metrics.roc_auc_score(Y_hologic_optimam_magviews, lr_duke_ge_radiomics_topfeats.predict_proba(X_hologic_optimam_magviews_normed[:, selected_feature_idx])[:,1])
	auc_optimam_magviews_nofs_ge_only = metrics.roc_auc_score(Y_ge_optimam_magviews, lr_duke_ge_radiomics_nofs.predict_proba(X_ge_optimam_magviews_normed)[:,1])
	auc_optimam_magviews_topfeats_ge_only = metrics.roc_auc_score(Y_ge_optimam_magviews, lr_duke_ge_radiomics_topfeats.predict_proba(X_ge_optimam_magviews_normed[:, selected_feature_idx])[:,1])
	print('OPTIMAM_MAGVIEWS NO Feature Selection round(AUC: %.3f (GE: %.3f + Hologic: %.3f); With TOP Feats: %.3f (GE: %.3f + Hologic: %.3f)' 
		%(round(auc_optimam_magviews_nofs, 3), round(auc_optimam_magviews_nofs_ge_only, 3), round(auc_optimam_magviews_nofs_hologic_only, 3), \
			round(auc_optimam_magviews_topfeats, 3), round(auc_optimam_magviews_topfeats_ge_only, 3), round(auc_optimam_magviews_topfeats_hologic_only, 3) ))
	##############################
	##############################
	## Magviews ALL AUC:
	optimam_magviews_unique_cases = np.unique([one_list[0] for one_list in info_optimam_magviews])
	Y_optimam_magviews_by_case = []
	proba_optimam_magviews_nofs_by_case_mean = []
	proba_optimam_magviews_nofs_by_case_max = []
	proba_optimam_magviews_topfeats_by_case_mean = []
	proba_optimam_magviews_topfeats_by_case_max = []
	for one_case in optimam_magviews_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_magviews) if one_list[0]==one_case]
		proba_optimam_magviews_nofs_by_case_mean.append(np.mean([proba_optimam_magviews_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_nofs_by_case_max.append(np.max([proba_optimam_magviews_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_topfeats_by_case_mean.append(np.mean([proba_optimam_magviews_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_magviews_topfeats_by_case_max.append(np.max([proba_optimam_magviews_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_magviews_by_case.append([info_optimam_magviews[ids][2] for ids in one_case_ids][0])

	auc_optimam_magviews_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_nofs_by_case_mean))
	auc_optimam_magviews_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_nofs_by_case_max))
	auc_optimam_magviews_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_topfeats_by_case_mean))
	auc_optimam_magviews_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_by_case), np.array(proba_optimam_magviews_topfeats_by_case_max))
	print('OPTIMAM_MAGVIEWS-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_magviews_nofs_by_case_mean, 3), round(auc_optimam_magviews_nofs_by_case_max, 3), \
			round(auc_optimam_magviews_topfeats_by_case_mean, 3), round(auc_optimam_magviews_topfeats_by_case_max,3)))
	##############################
	##############################
	## OPTIMAM Hologic
	proba_optimam_magviews_hologic_nofs = lr_duke_ge_radiomics_nofs.predict_proba(X_hologic_optimam_magviews_normed)[:,1]
	proba_optimam_magviews_hologic_topfeats = lr_duke_ge_radiomics_topfeats.predict_proba(X_hologic_optimam_magviews_normed[:, selected_feature_idx])[:,1]
	##
	optimam_magviews_hologic_unique_cases = np.unique([one_list[0] for one_list in info_optimam_magviews_hologic])
	Y_optimam_magviews_hologic_by_case = []
	proba_optimam_magviews_hologic_nofs_by_case_mean = []
	proba_optimam_magviews_hologic_nofs_by_case_max = []
	proba_optimam_magviews_hologic_topfeats_by_case_mean = []
	proba_optimam_magviews_hologic_topfeats_by_case_max = []
	for one_case in optimam_magviews_hologic_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_magviews_hologic) if one_list[0]==one_case]
		proba_optimam_magviews_hologic_nofs_by_case_mean.append(np.mean([proba_optimam_magviews_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_hologic_nofs_by_case_max.append(np.max([proba_optimam_magviews_hologic_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_hologic_topfeats_by_case_mean.append(np.mean([proba_optimam_magviews_hologic_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_magviews_hologic_topfeats_by_case_max.append(np.max([proba_optimam_magviews_hologic_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_magviews_hologic_by_case.append([info_optimam_magviews_hologic[ids][2] for ids in one_case_ids][0])

	auc_optimam_magviews_hologic_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_nofs_by_case_mean))
	auc_optimam_magviews_hologic_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_nofs_by_case_max))
	auc_optimam_magviews_hologic_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_topfeats_by_case_mean))
	auc_optimam_magviews_hologic_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_hologic_by_case), np.array(proba_optimam_magviews_hologic_topfeats_by_case_max))
	print('OPTIMAM_MAGVIEWS-Hologic Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_magviews_hologic_nofs_by_case_mean, 3), round(auc_optimam_magviews_hologic_nofs_by_case_max, 3), \
			round(auc_optimam_magviews_hologic_topfeats_by_case_mean, 3), round(auc_optimam_magviews_hologic_topfeats_by_case_max,3)))
	##############################
	##############################
	## OPTIMAM GE
	proba_optimam_magviews_ge_nofs = lr_duke_ge_radiomics_nofs.predict_proba(X_ge_optimam_magviews_normed)[:,1]
	proba_optimam_magviews_ge_topfeats = lr_duke_ge_radiomics_topfeats.predict_proba(X_ge_optimam_magviews_normed[:, selected_feature_idx])[:,1]
	##
	optimam_magviews_ge_unique_cases = np.unique([one_list[0] for one_list in info_optimam_magviews_ge])
	Y_optimam_magviews_ge_by_case = []
	proba_optimam_magviews_ge_nofs_by_case_mean = []
	proba_optimam_magviews_ge_nofs_by_case_max = []
	proba_optimam_magviews_ge_topfeats_by_case_mean = []
	proba_optimam_magviews_ge_topfeats_by_case_max = []
	for one_case in optimam_magviews_ge_unique_cases:
		one_case_ids = [idss for idss, one_list in enumerate(info_optimam_magviews_ge) if one_list[0]==one_case]
		proba_optimam_magviews_ge_nofs_by_case_mean.append(np.mean([proba_optimam_magviews_ge_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_ge_nofs_by_case_max.append(np.max([proba_optimam_magviews_ge_nofs[ids] for ids in one_case_ids]))
		proba_optimam_magviews_ge_topfeats_by_case_mean.append(np.mean([proba_optimam_magviews_ge_topfeats[ids] for ids in one_case_ids]))
		proba_optimam_magviews_ge_topfeats_by_case_max.append(np.max([proba_optimam_magviews_ge_topfeats[ids] for ids in one_case_ids]))
		Y_optimam_magviews_ge_by_case.append([info_optimam_magviews_ge[ids][2] for ids in one_case_ids][0])

	auc_optimam_magviews_ge_nofs_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_nofs_by_case_mean))
	auc_optimam_magviews_ge_nofs_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_nofs_by_case_max))
	auc_optimam_magviews_ge_topfeats_by_case_mean = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_topfeats_by_case_mean))
	auc_optimam_magviews_ge_topfeats_by_case_max = metrics.roc_auc_score(np.array(Y_optimam_magviews_ge_by_case), np.array(proba_optimam_magviews_ge_topfeats_by_case_max))
	print('OPTIMAM_MAGVIEWS-GE Testing AUCs (Model From DUKE), No FS Case MEAN: %.3f, Case MAX: %.3f; WITH Top Feats Case MEAN: %.3f, Case MAX: %.3f' \
		%(round(auc_optimam_magviews_ge_nofs_by_case_mean, 3), round(auc_optimam_magviews_ge_nofs_by_case_max, 3), \
			round(auc_optimam_magviews_ge_topfeats_by_case_mean, 3), round(auc_optimam_magviews_ge_topfeats_by_case_max,3)))

	# Train with DUKE-GE
	# --------- CV Only Logistic Regression Run 200 NO fs AUC: 0.660 +- 0.009; WITH fs AUC: 0.659 +- 0.009
	# OPTIMAM NO Feature Selection round(AUC: 0.602 (GE: 0.613 + Hologic: 0.598 + Philips: 0.631 + SIEMENS: 0.609); With TOP Feats: 0.613 (GE: 0.583 + Hologic: 0.614 + Philips: 0.612 + SIEMENS: 0.672)
	# OPTIMAM_MAGVIEWS NO Feature Selection round(AUC: 0.591 (GE: 0.687 + Hologic: 0.538); With TOP Feats: 0.612 (GE: 0.709 + Hologic: 0.561) 
	# OPTIMAM_MAGVIEWS-ALL Testing AUCs (Model From DUKE), No FS Case MEAN: 0.600, Case MAX: 0.606; WITH Top Feats Case MEAN: 0.629, Case MAX: 0.642
	# OPTIMAM_MAGVIEWS-Hologic Testing AUCs (Model From DUKE), No FS Case MEAN: 0.542, Case MAX: 0.560; WITH Top Feats Case MEAN: 0.578, Case MAX: 0.593
	# OPTIMAM_MAGVIEWS-GE Testing AUCs (Model From DUKE), No FS Case MEAN: 0.696, Case MAX: 0.689; WITH Top Feats Case MEAN: 0.727, Case MAX: 0.735

	

fpr_duke,tpr_duke,thres_duke = metrics.roc_curve(Y_duke, lr_radiomics_topfeats.predict_proba(X_duke_normed_subset)[:,1])
##
proba_nki_hologic_topfeats = lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1]
fpr_nki,tpr_nki,thres_nki = metrics.roc_curve(Y_hologic_nki,lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1])
##
proba_optimam_hologic_topfeats = lr_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1]
fpr_optimam,tpr_optimam,thres_optimam = metrics.roc_curve(Y_hologic_optimam,lr_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1])
####
idx_90tpr = np.where(tpr_duke==min(tpr_duke, key=lambda x:abs(x-0.9000)))[0][0]
thres_90tpr = thres_duke[idx_90tpr]
##
nki_cutoff_idx = np.where(proba_nki_hologic_topfeats>=thres_90tpr)
nki_cutoff_labels = Y_hologic_nki[nki_cutoff_idx]
nki_cutoff_tpr = np.where(nki_cutoff_labels==1)[0].shape[0]/np.where(Y_hologic_nki==1)[0].shape[0]
nki_cutoff_fpr = np.where(nki_cutoff_labels==0)[0].shape[0]/np.where(Y_hologic_nki==0)[0].shape[0]
##
optimam_cutoff_idx = np.where(proba_optimam_hologic_topfeats>=thres_90tpr)
optimam_cutoff_labels = Y_hologic_optimam[optimam_cutoff_idx]
optimam_cutoff_tpr = np.where(optimam_cutoff_labels==1)[0].shape[0]/np.where(Y_hologic_optimam==1)[0].shape[0]
optimam_cutoff_fpr = np.where(optimam_cutoff_labels==0)[0].shape[0]/np.where(Y_hologic_optimam==0)[0].shape[0]


_ = plt.plot(fpr_duke,tpr_duke,linewidth = 4, color = 'b',label = 'Self Train Duke, AUC = %.3f, Sens: %d%%, Spec: %d%%.'\
	%(round(metrics.roc_auc_score(Y_duke, lr_radiomics_topfeats.predict_proba(X_duke_normed_subset)[:,1]), 3), \
		round(tpr_duke[idx_90tpr]*100), round((1-fpr_duke[idx_90tpr])*100) ))
_= plt.plot(fpr_nki,tpr_nki,linewidth = 4, color = 'orange',label = 'NKI Test AUC = %.3f, Sens: %d%%, Spec: %d%%.'\
	%(round(metrics.roc_auc_score(Y_hologic_nki,lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1]), 3), \
		round(nki_cutoff_tpr*100), round((1-nki_cutoff_fpr)*100) ))
_ = plt.plot(fpr_optimam,tpr_optimam,linewidth = 4, color = 'red',label = 'OPTIMAM Test AUC = %.3f, Sens: %d%%, Spec: %d%%.'\
	%(round(metrics.roc_auc_score(Y_hologic_optimam,lr_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1]), 3), \
		round(optimam_cutoff_tpr*100), round((1-optimam_cutoff_fpr)*100) ))

_ = plt.plot(fpr[idx_90tpr], tpr[idx_90tpr], color = 'darkblue', marker = 'o', markersize = 10, alpha = 0.7)
_ = plt.plot(nki_cutoff_fpr, nki_cutoff_tpr, color = 'darkorange', marker = 'o', markersize = 10, alpha = 0.7)
_ = plt.plot(optimam_cutoff_fpr, optimam_cutoff_tpr, color = 'brown', marker = 'o', markersize = 10, alpha = 0.7)
_ = plt.xlabel('1-Specificity', fontsize= 16)
_ = plt.ylabel('Sensitivity', fontsize= 16)
_ = plt.legend(loc = 'upper left', fontsize = 14)
_ = plt.title('Cutoff results from Duke self train-test model', fontsize= 18)
plt.show()


idx_90fpr = np.where(fpr_duke==min(fpr_duke, key=lambda x:abs(x-0.1000)))[0][0]
thres_90fpr = thres_duke[idx_90fpr]
##
nki_cutoff_idx = np.where(proba_nki_hologic_topfeats>=thres_90fpr)
nki_cutoff_labels = Y_hologic_nki[nki_cutoff_idx]
nki_cutoff_tpr = np.where(nki_cutoff_labels==1)[0].shape[0]/np.where(Y_hologic_nki==1)[0].shape[0]
nki_cutoff_fpr = np.where(nki_cutoff_labels==0)[0].shape[0]/np.where(Y_hologic_nki==0)[0].shape[0]
##
optimam_cutoff_idx = np.where(proba_optimam_hologic_topfeats>=thres_90fpr)
optimam_cutoff_labels = Y_hologic_optimam[optimam_cutoff_idx]
optimam_cutoff_tpr = np.where(optimam_cutoff_labels==1)[0].shape[0]/np.where(Y_hologic_optimam==1)[0].shape[0]
optimam_cutoff_fpr = np.where(optimam_cutoff_labels==0)[0].shape[0]/np.where(Y_hologic_optimam==0)[0].shape[0]


_ = plt.plot(fpr_duke,tpr_duke,linewidth = 4, color = 'b',label = 'Self Train Duke, AUC = %.3f, Sens: %d%%, Spec: %d%%.'\
	%(round(metrics.roc_auc_score(Y_duke, lr_radiomics_topfeats.predict_proba(X_duke_normed_subset)[:,1]), 3), \
		round(tpr_duke[idx_90fpr]*100), round((1-fpr_duke[idx_90fpr])*100) ))
_= plt.plot(fpr_nki,tpr_nki,linewidth = 4, color = 'orange',label = 'NKI Test AUC = %.3f, Sens: %d%%, Spec: %d%%.'\
	%(round(metrics.roc_auc_score(Y_hologic_nki,lr_radiomics_topfeats.predict_proba(X_hologic_nki_normed[:, selected_feature_idx])[:,1]), 3), \
		round(nki_cutoff_tpr*100), round((1-nki_cutoff_fpr)*100) ))
_ = plt.plot(fpr_optimam,tpr_optimam,linewidth = 4, color = 'red',label = 'OPTIMAM Test AUC = %.3f, Sens: %d%%, Spec: %d%%.'\
	%(round(metrics.roc_auc_score(Y_hologic_optimam,lr_radiomics_topfeats.predict_proba(X_hologic_optimam_normed[:, selected_feature_idx])[:,1]), 3), \
		round(optimam_cutoff_tpr*100), round((1-optimam_cutoff_fpr)*100) ))

_ = plt.plot(fpr_duke[idx_90fpr], tpr_duke[idx_90fpr], color = 'darkblue', marker = 'o', markersize = 10, alpha = 0.7)
_ = plt.plot(nki_cutoff_fpr, nki_cutoff_tpr, color = 'darkorange', marker = 'o', markersize = 10, alpha = 0.7)
_ = plt.plot(optimam_cutoff_fpr, optimam_cutoff_tpr, color = 'brown', marker = 'o', markersize = 10, alpha = 0.7)
_ = plt.xlabel('1-Specificity', fontsize= 16)
_ = plt.ylabel('Sensitivity', fontsize= 16)
_ = plt.legend(loc = 'upper left', fontsize = 14)
_ = plt.title('Cutoff results from Duke self train-test model', fontsize= 18)
_=plt.adjust_subplots(top=0.95,bottom=0.08,left=0.095,right=0.935,hspace=0.2,wspace=0.2)

plt.show()

# The false positive rate is calculated as FP/FP+TN,
# The false negative rate  It’s calculated as FN/FN+TP

# The true positive rate (TPR ) TP/TP+FN. 

# The true negative rate (specificity) TN/TN+FP.

three_case_feature_info = openpyxl.Workbook()
duke_feature_sheet= three_case_feature_info.worksheets[0]
duke_feature_sheet.title = 'Duke'
for idx_now, [one_info, one_feat] in enumerate(zip(info_duke_ge, X_ge_duke)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		duke_feature_sheet.cell(row = idx_now + 1, column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		duke_feature_sheet.cell(row = idx_now + 1, column = one_added_column + len(one_info) + 1).value = one_sub_feat

for idx_now, [one_info, one_feat] in enumerate(zip(info_duke_hologic, X_hologic_duke)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		duke_feature_sheet.cell(row = idx_now + 1 + X_ge_duke.shape[0], column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		duke_feature_sheet.cell(row = idx_now + 1 + X_ge_duke.shape[0], column = one_added_column + len(one_info) + 1).value = one_sub_feat

####
nki_feature_sheet= three_case_feature_info.create_sheet(title = 'NKI')
for idx_now, [one_info, one_feat] in enumerate(zip(info_nki_ge, X_ge_nki)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		nki_feature_sheet.cell(row = idx_now + 1, column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		nki_feature_sheet.cell(row = idx_now + 1, column = one_added_column + len(one_info) + 1).value = one_sub_feat

for idx_now, [one_info, one_feat] in enumerate(zip(info_nki_hologic, X_hologic_nki)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		nki_feature_sheet.cell(row = idx_now + 1 + X_ge_nki.shape[0], column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		nki_feature_sheet.cell(row = idx_now + 1 + X_ge_nki.shape[0], column = one_added_column + len(one_info) + 1).value = one_sub_feat

for idx_now, [one_info, one_feat] in enumerate(zip(info_nki_siemens, X_siemens_nki)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		nki_feature_sheet.cell(row = idx_now + 1 + X_ge_nki.shape[0] + X_hologic_nki.shape[0], column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		nki_feature_sheet.cell(row = idx_now + 1 + X_ge_nki.shape[0] + X_hologic_nki.shape[0], column = one_added_column + len(one_info) + 1).value = one_sub_feat

###
####
optimam_feature_sheet= three_case_feature_info.create_sheet(title = 'OPTIMAM')
for idx_now, [one_info, one_feat] in enumerate(zip(info_optimam_hologic, X_hologic_optimam)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		optimam_feature_sheet.cell(row = idx_now + 1, column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		optimam_feature_sheet.cell(row = idx_now + 1, column = one_added_column + len(one_info) + 1).value = one_sub_feat

for idx_now, [one_info, one_feat] in enumerate(zip(info_optimam_ge, X_ge_optimam)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		optimam_feature_sheet.cell(row = idx_now + 1 + X_hologic_optimam.shape[0], column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		optimam_feature_sheet.cell(row = idx_now + 1 + X_hologic_optimam.shape[0], column = one_added_column + len(one_info) + 1).value = one_sub_feat

for idx_now, [one_info, one_feat] in enumerate(zip(info_optimam_siemens, X_siemens_optimam)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		optimam_feature_sheet.cell(row = idx_now + 1 + X_ge_optimam.shape[0] + X_hologic_optimam.shape[0], column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		optimam_feature_sheet.cell(row = idx_now + 1 + X_ge_optimam.shape[0] + X_hologic_optimam.shape[0], column = one_added_column + len(one_info) + 1).value = one_sub_feat

for idx_now, [one_info, one_feat] in enumerate(zip(info_optimam_philips, X_philips_optimam)):
	for one_sub_column, one_sub_info in enumerate(one_info):
		optimam_feature_sheet.cell(row = idx_now + 1 + X_ge_optimam.shape[0] + X_hologic_optimam.shape[0] + X_siemens_optimam.shape[0], column = one_sub_column + 1).value = one_sub_info
	for one_added_column, one_sub_feat in enumerate(one_feat.tolist()):
		optimam_feature_sheet.cell(row = idx_now + 1 + X_ge_optimam.shape[0] + X_hologic_optimam.shape[0] + X_siemens_optimam.shape[0], column = one_added_column + len(one_info) + 1).value = one_sub_feat



three_case_feature_info.save(filename = '/home/rui/Documents/OPTIMAM/combined_three_sets_info_and_feature_lists.xlsx')