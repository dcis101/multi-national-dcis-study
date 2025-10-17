# install.packages("cvAUC") CvBMROC ROCKIT
# install.packages("pROC")
# install.packages("ROCR")
# install.packages("PRROC")
## See Last Session for Updated Figures with One iteration's set of ROCs
library(cvAUC)
library(pROC)
library(ROCR)
color_list = c("green4","red", "magenta", "blue","deepskyblue3","orangered")
csv_saved_path = "training_results/"

########################################################################################################
## DCIS-Rs 
options(digits=20)
csv_pred_with_top_feats_cv_only = read.csv(paste(csv_saved_path, "With_11TOP_features_from_CV_feature_selection_predictions.csv",sep = ""),header = F, sep = ",")
csv_labels_with_top_feats_cv_only = read.csv(paste(csv_saved_path, "With_11TOP_features_from_CV_feature_selection_labels.csv",sep = ""),header = F, sep = ",")
pred_with_top_feats_cv_only <-list()
labels_with_top_feats_cv_only <-list()
for (ii in c(1:nrow(csv_labels_with_top_feats_cv_only))){
  pred_with_top_feats_cv_only[[ii]]<-unlist(csv_pred_with_top_feats_cv_only[ii,])
  labels_with_top_feats_cv_only[[ii]]<-unlist(csv_labels_with_top_feats_cv_only[ii,])
}
out_modela_cv_only_with_top_feats <-cvAUC(pred_with_top_feats_cv_only,labels_with_top_feats_cv_only)
stat_with_top_feats_cv_only = ci.cvAUC(predictions = pred_with_top_feats_cv_only,labels = labels_with_top_feats_cv_only,confidence = 0.95)
info_with_top_feats_cv_only = paste("CV AUC: ",round(stat_with_top_feats_cv_only$cvAUC,3),", [95%CI: ",round(stat_with_top_feats_cv_only$ci[1],3),"-",round(stat_with_top_feats_cv_only$ci[2],3),"]",sep = "")
##########################################FINALIZED STATISTICS RESULTS##########################################################
err_auc_each_iteration_cli_only_nofs = abs(auc_and_ci_each_iteration_cli_only_nofs[,1] - mean(auc_and_ci_each_iteration_cli_only_nofs,1))
err_auc_each_iteration_cli_only_nofs_normed =(err_auc_each_iteration_cli_only_nofs - min(err_auc_each_iteration_cli_only_nofs))/(max(err_auc_each_iteration_cli_only_nofs) - min(err_auc_each_iteration_cli_only_nofs)) 
err_auc_each_iteration_cv_only_nofs = abs(auc_and_ci_each_iteration_cv_only_nofs[,1] - mean(auc_and_ci_each_iteration_cv_only_nofs,1))
err_auc_each_iteration_cv_only_nofs_normed = (err_auc_each_iteration_cv_only_nofs - min(err_auc_each_iteration_cv_only_nofs))/(max(err_auc_each_iteration_cv_only_nofs) - min(err_auc_each_iteration_cv_only_nofs))
err_auc_each_iteration_cv_and_cli_nofs = abs(auc_and_ci_each_iteration_cv_and_cli_nofs[,1] - mean(auc_and_ci_each_iteration_cv_and_cli_nofs,1))
err_auc_each_iteration_cv_and_cli_nofs_normed = (err_auc_each_iteration_cv_and_cli_nofs - min(err_auc_each_iteration_cv_and_cli_nofs))/(max(err_auc_each_iteration_cv_and_cli_nofs) - min(err_auc_each_iteration_cv_and_cli_nofs))
err_auc_each_iteration_with_top_feats_cv_only = abs(auc_and_ci_each_iteration_with_top_feats_cv_only[,1] - mean(auc_and_ci_each_iteration_with_top_feats_cv_only,1))
err_auc_each_iteration_with_top_feats_cv_only_normed = (err_auc_each_iteration_with_top_feats_cv_only - min(err_auc_each_iteration_with_top_feats_cv_only))/(max(err_auc_each_iteration_with_top_feats_cv_only) - min(err_auc_each_iteration_with_top_feats_cv_only))
#############
err_auc_each_iteration_three_selected_models = err_auc_each_iteration_cli_only_nofs_normed + err_auc_each_iteration_cv_and_cli_nofs_normed + err_auc_each_iteration_with_top_feats_cv_only_normed
iteration_selected = which(err_auc_each_iteration_three_selected_models==min(err_auc_each_iteration_three_selected_models))
#########
first_fold_selected = as.integer((iteration_selected-1)*5+1) ## Iteration = 134
##############################################################################
## DCIS-Rs
options(digits=20)
pred_with_top_feats_cv_only = as.list(as.data.frame(t(csv_pred_with_top_feats_cv_only[first_fold_selected:(first_fold_selected+4),])))
label_with_top_feats_cv_only = as.list(as.data.frame(t(csv_labels_with_top_feats_cv_only[first_fold_selected:(first_fold_selected+4),])))
stat_selected_iteration_with_top_feats_cv_only= ci.cvAUC(predictions =pred_with_top_feats_cv_only,labels = label_with_top_feats_cv_only,confidence = 0.95)
out_selected_iteration_with_top_feats_cv_only = cvAUC(pred_with_top_feats_cv_only,label_with_top_feats_cv_only)
#########################################################################################################
dev.off() # use until "null device 1" shows
plot(seq(from=0, to  = 1, by = 0.01),seq(from=0, to  = 1, by = 0.01),type="l",lty = 2,ann=FALSE,xaxt='n',yaxt='n')
plot(out_selected_iteration_with_top_feats_cv_only$perf,col = color_list[1], lty = 1, lwd = 4, avg = "vertical", add = TRUE, smooth = FALSE, xaxt = "n", yaxt = "n")
points(x = 0.14,y = 0.25,pch = 18,col = "gray",cex = 1) 
axis(side=1,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("1.0","0.9","0.8","0.7","0.6","0.5","0.4","0.3","0.2","0.1","0.0"),font.axis = 2.4)
axis(side=2,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 2.4)
info_selected_iteration_with_top_feats_cv_only = paste("DCIS-Rs cvAUC: ", round(stat_selected_iteration_with_top_feats_cv_only$cvAUC,3),", [95%CI: ", round(stat_selected_iteration_with_top_feats_cv_only$ci[1],3),"-", round(stat_selected_iteration_with_top_feats_cv_only$ci[2],3),"]", sep = "")
##################################FINAL TESTING On TESTING DATA######################################################
csv_train_on_duke_test_nl = read.csv('training_results/Train_on_us_test_on_nl.csv',header = T, sep = ",")
header_info = colnames(csv_train_on_duke_test_nl)
labels_overall = csv_train_on_duke_test_nl[, which(header_info=="labels")]
pred_testing_cv_only_with_fs = csv_train_on_duke_test_nl[, which(header_info=="pred.top.feats")]
perf_testing_cv_only_with_fs <- performance(prediction(pred_testing_cv_only_with_fs, labels_overall),"tpr","fpr")
tat_testing_cv_only_with_fs = roc(predictor = pred_testing_cv_only_with_fs,response = labels_overall,ci = TRUE)
info_testing_cv_only_with_fs = paste("DCIS-Rs AUC: ",round(stat_testing_cv_only_with_fs$auc,3),", [95%CI: ",round(stat_testing_cv_only_with_fs$ci,3)[1],"-",round(stat_testing_cv_only_with_fs$ci,3)[3],"]",sep = "")
dev.off() # use until "null device 1" shows
plot(seq(from=0, to  = 1, by = 0.01),seq(from=0, to  = 1, by = 0.01),type="l",lty = 2,ann=FALSE,xaxt='n',yaxt='n')
plot(perf_testing_cv_only_with_fs, col = color_list[4], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
points(x = 0.14,y = 0.23,pch = 18,col = "black",cex =  3)
axis(side=1,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("1.0","0.9","0.8","0.7","0.6","0.5","0.4","0.3","0.2","0.1","0.0"),font.axis = 3)
axis(side=2,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 3)

legend(0.4, 0.2, legend=c("Clinical Rules' Operating Point: (86%, 23%)",info_testing_cv_only_with_fs),
       col=c("black", color_list[6]), lty=c(NA,1,), pch=c(18,NA),cex=1.2, text.font=2.2, box.lty=2)
