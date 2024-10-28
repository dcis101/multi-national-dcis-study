# install.packages("cvAUC") CvBMROC ROCKIT
# install.packages("pROC")
# install.packages("ROCR")
# install.packages("PRROC")
## See Last Session for Updated Figures with One iteration's set of ROCs
library(cvAUC)
library(pROC)
library(ROCR)
color_list = c("green4","red", "magenta", "blue","deepskyblue3","orangered")
csv_saved_path = "saved_logs_predictions_and_labels/"

⁨########################################################################################################
## DCIS-C
csv_predictions_cli_only_nofs = read.csv(paste(csv_saved_path, "CLINICAL_FOUR_only_Predictions_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
csv_labels_cli_only_nofs = read.csv(paste(csv_saved_path, "CLINICAL_FOUR_only_Labels_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
pred_cli_only_nofs <- list()
labels_cli_only_nofs <- list()
for (ii in c(1:nrow(csv_labels_cli_only_nofs))){
  pred_cli_only_nofs[[ii]]<-unlist(csv_predictions_cli_only_nofs[ii,])
  labels_cli_only_nofs[[ii]]<-unlist(csv_labels_cli_only_nofs[ii,])
}
out_cli_only_nofs<-cvAUC(pred_cli_only_nofs,labels_cli_only_nofs)
########################################################################################################
## DCIS-R
options(digits=20)
csv_predictions_modela_cv_only_nofs = read.csv(paste(csv_saved_path, "CV_ONLY_NO-FS_Predictions_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
csv_labels_modela_cv_only_nofs = read.csv(paste(csv_saved_path, "CV_ONLY_Labels_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
pred_cv_nofs <- list()
labels_cv_nofs <- list()
for (ii in c(1:nrow(csv_labels_modela_cv_only_nofs))){
  pred_cv_nofs[[ii]]<-unlist(csv_predictions_modela_cv_only_nofs[ii,])
  labels_cv_nofs[[ii]]<-unlist(csv_labels_modela_cv_only_nofs[ii,])
}
out_modela_cv_only_nofs<-cvAUC(pred_cv_nofs,labels_cv_nofs)
########################################################################################################
## DCIS-RC
options(digits=20)
csv_predictions_modela_cv_and_cli_nofs = read.csv(paste(csv_saved_path, "CV_AND_Clinical_NO-FS_Predictions_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
csv_labels_modela_cv_and_cli_nofs = read.csv(paste(csv_saved_path, "CV_AND_Clinical_Labels_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
pred_cv_and_cli_nofs <- list()
labels_cv_and_cli_nofs <- list()
for (ii in c(1:nrow(csv_labels_modela_cv_and_cli_nofs))){
  pred_cv_and_cli_nofs[[ii]]<-unlist(csv_predictions_modela_cv_and_cli_nofs[ii,])
  labels_cv_and_cli_nofs[[ii]]<-unlist(csv_labels_modela_cv_and_cli_nofs[ii,])
}

out_modela_cv_and_cli_nofs<-cvAUC(pred_cv_and_cli_nofs,labels_cv_and_cli_nofs)
########################################################################################################
## DCIS-Rs
options(digits=20)
csv_pred_with_fs_cv_only = read.csv(paste(csv_saved_path, "CV_ONLY_WITH-FS_Predictions_with_5folds_200iterations.csv",sep = ""),header = F, sep = ",")
csv_labels_with_fs_cv_only = read.csv(paste(csv_saved_path, "CV_ONLY_Labels_with_5folds_200iterations.csv",sep = ""),header = F, sep = ",")
pred_with_fs_cv_only <-list()
labels_with_fs_cv_only <-list()
for (ii in c(1:nrow(csv_labels_with_fs_cv_only))){
  pred_with_fs_cv_only[[ii]]<-unlist(csv_pred_with_fs_cv_only[ii,])
  labels_with_fs_cv_only[[ii]]<-unlist(csv_labels_with_fs_cv_only[ii,])
}
out_modela_cv_only_with_fs <-cvAUC(pred_with_fs_cv_only,labels_with_fs_cv_only)
########################################################################################################
## DCIS-Rs(top)
csv_pred_with_top_feats_cv_only = read.csv(paste(csv_saved_path, "With_11TOP_features_from_CV_feature_selection_predictions.csv",sep = ""),header = F, sep = ",")
csv_labels_with_top_feats_cv_only = read.csv(paste(csv_saved_path, "With_11TOP_features_from_CV_feature_selection_labels.csv",sep = ""),header = F, sep = ",")
pred_with_top_feats_cv_only <-list()
labels_with_top_feats_cv_only <-list()
for (ii in c(1:nrow(csv_labels_with_top_feats_cv_only))){
  pred_with_top_feats_cv_only[[ii]]<-unlist(csv_pred_with_top_feats_cv_only[ii,])
  labels_with_top_feats_cv_only[[ii]]<-unlist(csv_labels_with_top_feats_cv_only[ii,])
}
out_modela_cv_only_with_top_feats <-cvAUC(pred_with_top_feats_cv_only,labels_with_top_feats_cv_only)
########################################################################################################

# ###########################DRAW CROSS-Validated Averaged ROCs##########################################
# color_list = c("magenta", "slateblue", "green4", "blue","red", "deepskyblue3")
# dev.off() # use until "null device 1" shows

# plot(seq(from=0, to  = 1, by = 0.01),seq(from=0, to  = 1, by = 0.01),type="l",lty = 2,ann=FALSE)
# plot(out_modela_cv_only_nofs$perf,col = color_list[1],lty=1,lwd = 4,avg="vertical",add=TRUE,smooth=FALSE)
# plot(out_modela_cv_and_cli_nofs$perf,col = color_list[2],lty=1,lwd = 4,avg="vertical",add=TRUE,smooth=FALSE)
# plot(out_modeld_cv_nofs$perf,col = color_list[3],lty=1,lwd = 4,avg="vertical",add=TRUE,smooth=FALSE)
# plot(out_modela_cv_only_with_fs$perf,col = color_list[4],lty = 1,lwd = 5, avg = "vertical",add=TRUE,smooth=FALSE)
# plot(out_modela_cv_and_cli_with_fs$perf,col = color_list[5],lty = 1,lwd = 5, avg = "vertical",add=TRUE,smooth=FALSE)
## plot(out_modela_cv_only_with_top_feats$perf,col = color_list[6],lty = 1,lwd = 5, avg = "vertical",add=TRUE,smooth=FALSE)

# ###########CI Info##########################
# stat_cv_nofs = ci.cvAUC(predictions = pred_cv_nofs, labels = labels_cv_nofs, confidence = 0.95)
# stat_cv_and_cli_nofs = ci.cvAUC(predictions = pred_cv_and_cli_nofs, labels = labels_cv_and_cli_nofs, confidence = 0.95)
# stat_cv_modeld = ci.cvAUC(predictions = pred_cv_modeld, labels = labels_cv_modeld, confidence = 0.95)
# stat_with_fs_cv_only = ci.cvAUC(predictions = pred_with_fs_cv_only,labels = labels_with_fs_cv_only,confidence = 0.95)
# stat_with_fs_cv_and_cli = ci.cvAUC(predictions = pred_with_fs_cv_and_cli,labels = labels_with_fs_cv_and_cli,confidence = 0.95)
# stat_with_top_feats_cv_only = ci.cvAUC(predictions = pred_with_top_feats_cv_only,labels = labels_with_top_feats_cv_only,confidence = 0.95)
# #######
# info_cv_nofs = paste("Model A:Radiomics Only Without Feature Selection AUC: ",round(stat_cv_nofs$cvAUC,3),", [95%CI: ",round(stat_cv_nofs$ci[1],3),"-",round(stat_cv_nofs$ci[2],3),"]",sep = "")
# info_cv_and_cli_nofs = paste("Model A:Radiomics + Clinical Features Without Feature Selection AUC: ",round(stat_cv_and_cli_nofs$cvAUC,3),", [95%CI: ",round(stat_cv_and_cli_nofs$ci[1],3),"-",round(stat_cv_and_cli_nofs$ci[2],3),"]",sep = "")
# info_cv_modeld = paste("Model D:Radiomics Only Without Feature Selection AUC: ",round(stat_cv_modeld$cvAUC,3),", [95%CI: ",round(stat_cv_modeld$ci[1],3),"-",round(stat_cv_modeld$ci[2],3),"]",sep = "")
# info_with_fs_cv_only = paste("Model A:Radiomics Only With Feature Selection AUC: ",round(stat_with_fs_cv_only$cvAUC,3),", [95%CI: ",round(stat_with_fs_cv_only$ci[1],3),"-",round(stat_with_fs_cv_only$ci[2],3),"]",sep = "")
# info_with_fs_cv_and_cli = paste("Model A:Radiomics + Clinical Features With Feature Selection AUC: ",round(stat_with_fs_cv_and_cli$cvAUC,3),", [95%CI: ",round(stat_with_fs_cv_and_cli$ci[1],3),"-",round(stat_with_fs_cv_and_cli$ci[2],3),"]",sep = "")

# legend(0.2, 0.15, legend=c(info_cv_nofs, info_cv_and_cli_nofs, info_cv_modeld, info_with_fs_cv_only, info_with_fs_cv_and_cli),
#        col=c(color_list[1],color_list[2],color_list[3],color_list[4],color_list[5]), lty=1, cex=1, text.font=2.4, box.lty=2)
# title('Model A ROCs of with Radiomics and Clinical Features',cex.main = 2.2, line = 1)

###########CI Info##########################
stat_cv_nofs = ci.cvAUC(predictions = pred_cv_nofs, labels = labels_cv_nofs, confidence = 0.95)
stat_cv_and_cli_nofs = ci.cvAUC(predictions = pred_cv_and_cli_nofs, labels = labels_cv_and_cli_nofs, confidence = 0.95)
stat_cv_modeld = ci.cvAUC(predictions = pred_cv_modeld, labels = labels_cv_modeld, confidence = 0.95)
stat_with_fs_cv_only = ci.cvAUC(predictions = pred_with_fs_cv_only,labels = labels_with_fs_cv_only,confidence = 0.95)
stat_with_top_feats_cv_only = ci.cvAUC(predictions = pred_with_top_feats_cv_only,labels = labels_with_top_feats_cv_only,confidence = 0.95)
stat_cli_only_nofs = ci.cvAUC(predictions = pred_cli_only_nofs, labels = labels_cli_only_nofs, confidence = 0.95)

dev.off() # use until "null device 1" shows

plot(seq(from=0, to  = 1, by = 0.01),seq(from=0, to  = 1, by = 0.01),type="l",lty = 2,ann=FALSE,xaxt='n',yaxt='n')
plot(out_modela_cv_only_nofs$perf,col = color_list[1],lty=1,lwd = 4,avg="vertical",add=TRUE,smooth=FALSE,xaxt='n',yaxt='n')
plot(out_modela_cv_only_with_fs$perf,col = color_list[4],lty = 1,lwd = 5, avg = "vertical",add=TRUE,smooth=FALSE,xaxt='n',yaxt='n')
plot(out_modela_cv_and_cli_nofs$perf,col = color_list[2],lty=1,lwd = 4,avg="vertical",add=TRUE,smooth=FALSE,xaxt='n',yaxt='n')
# plot(out_modeld_cv_nofs$perf,col = color_list[3],lty=1,lwd = 4,avg="vertical",add=TRUE,smooth=FALSE,xaxt='n',yaxt='n')
points(x = 0.14,y = 0.25,pch = 18,col = "gray",cex = 0.6) # 
#######
info_cv_nofs = paste("DCIS-R Averaged AUC: ",round(stat_cv_nofs$cvAUC,3),", [95%CI: ",round(stat_cv_nofs$ci[1],3),"-",round(stat_cv_nofs$ci[2],3),"]",sep = "")
info_cv_and_cli_nofs = paste("DCIS-RC Averaged AUC: ",round(stat_cv_and_cli_nofs$cvAUC,3),", [95%CI: ",round(stat_cv_and_cli_nofs$ci[1],3),"-",round(stat_cv_and_cli_nofs$ci[2],3),"]",sep = "")
info_cv_modeld = paste("ADHIDC-R Averaged AUC: ",round(stat_cv_modeld$cvAUC,3),", [95%CI: ",round(stat_cv_modeld$ci[1],3),"-",round(stat_cv_modeld$ci[2],3),"]",sep = "")
info_with_fs_cv_only = paste("DCIS-Rs Averaged AUC: ",round(stat_with_fs_cv_only$cvAUC,3),", [95%CI: ",round(stat_with_fs_cv_only$ci[1],3),"-",round(stat_with_fs_cv_only$ci[2],3),"]",sep = "")
info_with_top_feats_cv_only = paste("DCIS-Rs(top) Averaged AUC: ",round(stat_with_top_feats_cv_only$cvAUC,3),", [95%CI: ",round(stat_with_top_feats_cv_only$ci[1],3),"-",round(stat_with_top_feats_cv_only$ci[2],3),"]",sep = "")

axis(side=1,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("1.0","0.9","0.8","0.7","0.6","0.5","0.4","0.3","0.2","0.1","0.0"),font.axis = 2.4)
axis(side=2,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 2.4)

legend(0.3, 0.17, legend=c(info_cv_nofs, info_with_fs_cv_only, info_cv_and_cli_nofs, info_cv_modeld,"Clinical Rules"),
       col=c(color_list[1],color_list[4],color_list[2],color_list[3],"black"), lty=c(1,1,1,1,NA), pch=c(NA,NA,NA,NA,18),cex=1, text.font=2.2, box.lty=2)
title('Cross-Validated ROCs of Training Set',cex.main = 1.6, line = 2.2,xlab = "Specificity",ylab = "Sensitivity", cex.lab = 1.6)
#########
plot(perf_train_cv_only_modeld, col = color_list[3], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
legend(0.25, 0.2, legend=c(info_cv_nofs, info_with_fs_cv_only, info_cv_and_cli_nofs,info_train_cv_only_modeld),
       col=c(color_list[1],color_list[4],color_list[2],color_list[3],"black"), lty=1, cex=1.2, text.font=2, box.lty=2)
title('Cross-Validated ROCs of Training Set',cex.main = 1.6, line = 2.2,xlab = "Specificity",ylab = "Sensitivity", cex.lab = 1.6)


#################################################################################################################################################
##########################################FINALIZED STATISTICS RESULTS##########################################################
#################################################################################################################################################
csv_saved_path = "saved_logs_predictions_and_labels/"
library(cvAUC)
library(pROC)
library(ROCR)
color_list = c("green4","red", "magenta", "blue","deepskyblue3","orangered")
########################################################################################################
## DCIS-C
csv_predictions_cli_only_nofs = read.csv(paste(csv_saved_path, "CLINICAL_FOUR_only_Predictions_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
csv_labels_cli_only_nofs = read.csv(paste(csv_saved_path, "CLINICAL_FOUR_only_Labels_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
auc_and_ci_each_iteration_cli_only_nofs = matrix(NA,as.integer(nrow(csv_labels_cli_only_nofs)/5),3)
for (iter in c(1:as.integer(nrow(csv_labels_cli_only_nofs)/5))){
  first_fold = as.integer((iter-1)*5+1)
  one_iter_stat= ci.cvAUC(predictions = as.list(as.data.frame(t(csv_predictions_cli_only_nofs[first_fold:(first_fold+4),]))),labels = as.list(as.data.frame(t(csv_labels_cli_only_nofs[first_fold:(first_fold+4),]))),confidence = 0.95)
  auc_and_ci_each_iteration_cli_only_nofs[iter,] = c(round(one_iter_stat$cvAUC,3),round(one_iter_stat$ci[1],3),round(one_iter_stat$ci[2],3))
}
########################################################################################################
## DCIS-R
options(digits=20)
csv_predictions_modela_cv_only_nofs = read.csv(paste(csv_saved_path, "CV_ONLY_NO-FS_Predictions_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
csv_labels_modela_cv_only_nofs = read.csv(paste(csv_saved_path, "CV_ONLY_Labels_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
auc_and_ci_each_iteration_cv_only_nofs = matrix(NA,as.integer(nrow(csv_labels_modela_cv_only_nofs)/5),3)
for (iter in c(1:as.integer(nrow(csv_labels_modela_cv_only_nofs)/5))){
  first_fold = as.integer((iter-1)*5+1)
  one_iter_stat= ci.cvAUC(predictions = as.list(as.data.frame(t(csv_predictions_modela_cv_only_nofs[first_fold:(first_fold+4),]))),labels = as.list(as.data.frame(t(csv_labels_modela_cv_only_nofs[first_fold:(first_fold+4),]))),confidence = 0.95)
  auc_and_ci_each_iteration_cv_only_nofs[iter,] = c(round(one_iter_stat$cvAUC,3),round(one_iter_stat$ci[1],3),round(one_iter_stat$ci[2],3))
}
########################################################################################################
## DCIS-RC
options(digits=20)
csv_predictions_modela_cv_and_cli_nofs = read.csv(paste(csv_saved_path, "CV_AND_Clinical_NO-FS_Predictions_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
csv_labels_modela_cv_and_cli_nofs = read.csv(paste(csv_saved_path, "CV_AND_Clinical_Labels_with_5folds_200iterations.csv",sep = ""),header = F,sep = ",")
auc_and_ci_each_iteration_cv_and_cli_nofs = matrix(NA,as.integer(nrow(csv_labels_modela_cv_and_cli_nofs)/5),3)
for (iter in c(1:as.integer(nrow(csv_labels_modela_cv_and_cli_nofs)/5))){
  first_fold = as.integer((iter-1)*5+1)
  one_iter_stat= ci.cvAUC(predictions = as.list(as.data.frame(t(csv_predictions_modela_cv_and_cli_nofs[first_fold:(first_fold+4),]))),labels = as.list(as.data.frame(t(csv_labels_modela_cv_and_cli_nofs[first_fold:(first_fold+4),]))),confidence = 0.95)
  auc_and_ci_each_iteration_cv_and_cli_nofs[iter,] = c(round(one_iter_stat$cvAUC,3),round(one_iter_stat$ci[1],3),round(one_iter_stat$ci[2],3))
}
########################################################################################################
## DCIS-Rs(top)
csv_pred_with_top_feats_cv_only = read.csv(paste(csv_saved_path, "With_11TOP_features_from_CV_feature_selection_predictions.csv",sep = ""),header = F, sep = ",")
csv_labels_with_top_feats_cv_only = read.csv(paste(csv_saved_path, "With_11TOP_features_from_CV_feature_selection_labels.csv",sep = ""),header = F, sep = ",")
auc_and_ci_each_iteration_with_top_feats_cv_only = matrix(NA,as.integer(nrow(csv_labels_with_top_feats_cv_only)/5),3)
for (iter in c(1:as.integer(nrow(csv_labels_with_top_feats_cv_only)/5))){
  first_fold = as.integer((iter-1)*5+1)
  one_iter_stat= ci.cvAUC(predictions = as.list(as.data.frame(t(csv_pred_with_top_feats_cv_only[first_fold:(first_fold+4),]))),labels = as.list(as.data.frame(t(csv_labels_with_top_feats_cv_only[first_fold:(first_fold+4),]))),confidence = 0.95)
  auc_and_ci_each_iteration_with_top_feats_cv_only[iter,] = c(round(one_iter_stat$cvAUC,3),round(one_iter_stat$ci[1],3),round(one_iter_stat$ci[2],3))
}

#############
#############
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
## DCIS-C
options(digits=20)
pred_cli_only_nofs = as.list(as.data.frame(t(csv_predictions_cli_only_nofs[first_fold_selected:(first_fold_selected+4),])))
labels_cli_only_nofs = as.list(as.data.frame(t(csv_labels_cli_only_nofs[first_fold_selected:(first_fold_selected+4),])))
stat_selected_iteration_cli_only_nofs = ci.cvAUC(predictions = pred_cli_only_nofs,labels = labels_cli_only_nofs, confidence = 0.95)
out_selected_iteration_cli_only_nofs<-cvAUC(pred_cli_only_nofs,labels_cli_only_nofs)
########################################################################################################
## DCIS-R
options(digits=20)
pred_cv_only_nofs = as.list(as.data.frame(t(csv_predictions_modela_cv_only_nofs[first_fold_selected:(first_fold_selected+4),])))
label_cv_only_nofs = as.list(as.data.frame(t(csv_labels_modela_cv_only_nofs[first_fold_selected:(first_fold_selected+4),])))
stat_selected_iteration_cv_only_nofs= ci.cvAUC(predictions =pred_cv_only_nofs,labels = label_cv_only_nofs,confidence = 0.95)
out_selected_iteration_cv_only_nofs = cvAUC(pred_cv_only_nofs,label_cv_only_nofs)
########################################################################################################
## DCIS-RC
options(digits=20)
pred_cv_and_cli_nofs = as.list(as.data.frame(t(csv_predictions_modela_cv_and_cli_nofs[first_fold_selected:(first_fold_selected+4),])))
label_cv_and_cli_nofs = as.list(as.data.frame(t(csv_labels_modela_cv_and_cli_nofs[first_fold_selected:(first_fold_selected+4),])))
stat_selected_iteration_cv_and_cli_nofs= ci.cvAUC(predictions =pred_cv_and_cli_nofs,labels = label_cv_and_cli_nofs,confidence = 0.95)
out_selected_iteration_cv_and_cli_nofs = cvAUC(pred_cv_and_cli_nofs,label_cv_and_cli_nofs)
########################################################################################################
## DCIS-Rs
options(digits=20)
pred_with_top_feats_cv_only = as.list(as.data.frame(t(csv_pred_with_top_feats_cv_only[first_fold_selected:(first_fold_selected+4),])))
label_with_top_feats_cv_only = as.list(as.data.frame(t(csv_labels_with_top_feats_cv_only[first_fold_selected:(first_fold_selected+4),])))
stat_selected_iteration_with_top_feats_cv_only= ci.cvAUC(predictions =pred_with_top_feats_cv_only,labels = label_with_top_feats_cv_only,confidence = 0.95)
out_selected_iteration_with_top_feats_cv_only = cvAUC(pred_with_top_feats_cv_only,label_with_top_feats_cv_only)
#########################################################################################################
dev.off() # use until "null device 1" shows
plot(seq(from=0, to  = 1, by = 0.01),seq(from=0, to  = 1, by = 0.01),type="l",lty = 2,ann=FALSE,xaxt='n',yaxt='n')
plot(out_selected_iteration_cli_only_nofs$perf,col = color_list[6], lty = 1, lwd = 4, avg = "vertical", add = TRUE, smooth = FALSE, xaxt = "n", yaxt = "n")
plot(out_selected_iteration_cv_only_nofs$perf,col = color_list[2], lty = 1, lwd = 4, avg = "vertical", add = TRUE, smooth = FALSE, xaxt = "n", yaxt = "n")
plot(out_selected_iteration_cv_and_cli_nofs$perf,col = color_list[4], lty = 1, lwd = 4, avg = "vertical", add = TRUE, smooth = FALSE, xaxt = "n", yaxt = "n")
plot(out_selected_iteration_with_top_feats_cv_only$perf,col = color_list[1], lty = 1, lwd = 4, avg = "vertical", add = TRUE, smooth = FALSE, xaxt = "n", yaxt = "n")

points(x = 0.14,y = 0.25,pch = 18,col = "gray",cex = 1) # out of 400 cases, 12 cases don't have ER or PR or NG info.
axis(side=1,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("1.0","0.9","0.8","0.7","0.6","0.5","0.4","0.3","0.2","0.1","0.0"),font.axis = 2.4)
axis(side=2,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 2.4)
##
#######
info_selected_iteration_cli_only_nofs = paste("DCIS-C cvAUC: ", round(stat_selected_iteration_cli_only_nofs$cvAUC,3),", [95%CI: ", round(stat_selected_iteration_cli_only_nofs$ci[1],3),"-", round(stat_selected_iteration_cli_only_nofs$ci[2],3),"]", sep = "")
info_selected_iteration_cv_only_nofs = paste("DCIS-R cvAUC: ", round(stat_selected_iteration_cv_only_nofs$cvAUC,3),", [95%CI: ", round(stat_selected_iteration_cv_only_nofs$ci[1],3),"-", round(stat_selected_iteration_cv_only_nofs$ci[2],3),"]", sep = "")
info_selected_iteration_cv_and_cli_nofs = paste("DCIS-RC cvAUC: ", round(stat_selected_iteration_cv_and_cli_nofs$cvAUC,3),", [95%CI: ", round(stat_selected_iteration_cv_and_cli_nofs$ci[1],3),"-", round(stat_selected_iteration_cv_and_cli_nofs$ci[2],3),"]", sep = "")
info_selected_iteration_with_top_feats_cv_only = paste("DCIS-Rs cvAUC: ", round(stat_selected_iteration_with_top_feats_cv_only$cvAUC,3),", [95%CI: ", round(stat_selected_iteration_with_top_feats_cv_only$ci[1],3),"-", round(stat_selected_iteration_with_top_feats_cv_only$ci[2],3),"]", sep = "")
# info_train_cv_only_modeld
#######
legend(0.32, 0.3, legend=c(info_selected_iteration_cli_only_nofs, info_selected_iteration_cv_only_nofs,info_selected_iteration_cv_and_cli_nofs, info_selected_iteration_with_top_feats_cv_only,info_selected_iteration_cv_and_cli_top_feats,info_train_cv_only_modeld),
       col=c(color_list[6],color_list[2],color_list[4],color_list[1],color_list[2],color_list[3]), lty=1, cex=1.4, text.font=2.2, box.lty=2)


legend(0.32, 0.15, legend=c(info_selected_iteration_cli_only_nofs, info_selected_iteration_cv_and_cli_nofs, info_selected_iteration_with_top_feats_cv_only,info_train_cv_only_modeld),
       col=c(color_list[6],color_list[4],color_list[1],color_list[3]), lty=1, cex=1.4, text.font=2.2, box.lty=2)
title('Training Set ROC Curves from One Iteration',cex.main = 1.6, line = 2.2,xlab = "Specificity",ylab = "Sensitivity", cex.lab = 1.8)
################################################################################
################################4 Models Test ROC and PR#####################################
##################################FINAL TESTING On TESTING DATA######################################################

csv_saved_path = "saved_logs_predictions_and_labels/"

library(cvAUC)
library(ROCR)
library(pROC)

options(digits=20)
csv_final_testing_models_predictions = read.csv(paste(csv_saved_path, "Final_Test300_from_train400_pred_and_labels_4models_cvtop11_109CV_4Cli.csv",sep = ""),header = T, sep = ",")
header_info = colnames(csv_final_testing_models_predictions)
labels_overall = csv_final_testing_models_predictions[, which(header_info=="labels")]

pred_testing_cv_only_nofs = csv_final_testing_models_predictions[, which(header_info=="pred.CV")]
pred_testing_cv_and_cli_nofs = csv_final_testing_models_predictions[, which(header_info=="pred.CV.plus.clinical")]
pred_testing_cv_only_with_fs = csv_final_testing_models_predictions[, which(header_info=="pred.top.feats")]
pred_testing_cv_only_modeld = csv_final_testing_models_predictions[, which(header_info=="pred.ModelD.lr")]
# pred_testing_cv_and_cli_fs = csv_final_testing_models_predictions[, which(header_info=="pred.CV")]
pred_testing_cli_only = csv_final_testing_models_predictions[, which(header_info=="pred.cli.only")]
################
perf_testing_cv_only_nofs <- performance(prediction(pred_testing_cv_only_nofs, labels_overall),"tpr","fpr")
perf_testing_cv_and_cli_nofs <- performance(prediction(pred_testing_cv_and_cli_nofs, labels_overall),"tpr","fpr")
perf_testing_cv_only_with_fs <- performance(prediction(pred_testing_cv_only_with_fs, labels_overall),"tpr","fpr")
perf_testing_cv_only_modeld <- performance(prediction(pred_testing_cv_only_modeld, labels_overall),"tpr","fpr")
perf_testing_cli_only <- performance(prediction(pred_testing_cli_only, labels_overall),"tpr","fpr")
###############
stat_testing_cv_only_nofs = roc(predictor = pred_testing_cv_only_nofs,response = labels_overall,ci = TRUE)
stat_testing_cv_and_cli_nofs = roc(predictor = pred_testing_cv_and_cli_nofs,response = labels_overall,ci = TRUE)
stat_testing_cv_only_with_fs = roc(predictor = pred_testing_cv_only_with_fs,response = labels_overall,ci = TRUE)
stat_testing_cv_only_modeld = roc(predictor = pred_testing_cv_only_modeld,response = labels_overall,ci = TRUE)
stat_testing_cli_only = roc(predictor = pred_testing_cli_only,response = labels_overall,ci = TRUE)
###############
info_testing_cv_only_nofs = paste("DCIS-R AUC: ",round(stat_testing_cv_only_nofs$auc,3),", [95%CI: ",round(stat_testing_cv_only_nofs$ci,3)[1],"-",round(stat_testing_cv_only_nofs$ci,3)[3],"]",sep = "")
info_testing_cv_and_cli_nofs = paste("DCIS-RC AUC: ",round(stat_testing_cv_and_cli_nofs$auc,3),", [95%CI: ",round(stat_testing_cv_and_cli_nofs$ci,3)[1],"-",round(stat_testing_cv_and_cli_nofs$ci,3)[3],"]",sep = "")
info_testing_cv_only_with_fs = paste("DCIS-Rs AUC: ",round(stat_testing_cv_only_with_fs$auc,3),", [95%CI: ",round(stat_testing_cv_only_with_fs$ci,3)[1],"-",round(stat_testing_cv_only_with_fs$ci,3)[3],"]",sep = "")
info_testing_cli_only = paste("DCIS-C AUC: ",round(stat_testing_cli_only$auc,3),", [95%CI: ",round(stat_testing_cli_only$ci,3)[1],"-",round(stat_testing_cli_only$ci,3)[3],"]",sep = "")
#############################################
library(ROCR)
library(pROC)
## Compare two ROCs/models, this command will give output on p-values and AUCs for two curves
roc.test(response = labels_overall,predictor1 = pred_testing_cli_only,predictor2 = pred_testing_cv_and_cli_nofs,ternative = c("two.sided"))
## or your can do roc_comp = roc.test(...), then call roc_comp$p to get p-value

dev.off() # use until "null device 1" shows
plot(seq(from=0, to  = 1, by = 0.01),seq(from=0, to  = 1, by = 0.01),type="l",lty = 2,ann=FALSE,xaxt='n',yaxt='n')
# plot(perf_testing_cv_only_nofs, col = color_list[1], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
plot(perf_testing_cli_only, col = color_list[6], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
plot(perf_testing_cv_only_with_fs, col = color_list[4], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
plot(perf_testing_cv_and_cli_nofs, col = color_list[3], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
plot(perf_testing_cv_only_modeld, col = color_list[1], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
points(x = 0.14,y = 0.23,pch = 18,col = "black",cex =  3) # out of 300 cases, 2 cases don't have ER or PR or NG info

axis(side=1,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("1.0","0.9","0.8","0.7","0.6","0.5","0.4","0.3","0.2","0.1","0.0"),font.axis = 3)
axis(side=2,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 3)

legend(0.4, 0.2, legend=c("Clinical Rules' Operating Point: (86%, 23%)",info_testing_cli_only,info_testing_cv_only_with_fs, info_testing_cv_and_cli_nofs, info_testing_cv_only_modeld),
       col=c("black", color_list[6],color_list[4],color_list[3],color_list[1]), lty=c(NA,1,1,1,1), pch=c(18,NA,NA,NA,NA),cex=1.2, text.font=2.2, box.lty=2)

legend(0.28, 0.18, legend=c(info_testing_cv_only_nofs, info_testing_cv_only_with_fs, info_testing_cv_and_cli_nofs, info_testing_cv_only_modeld),
       col=c(color_list[1],color_list[4],color_list[2],color_list[3]), lty=1,cex=1.5, text.font=2.2, box.lty=2)

legend(0.3, 0.24, legend=c(info_testing_cv_only_with_fs, info_testing_cv_and_cli_nofs, info_testing_cv_only_modeld),
       col=c(color_list[4],color_list[1],color_list[3]), lty=1,cex=1.5, text.font=2.2, box.lty=2)

legend(0.33, 0.3, legend=c(info_testing_cli_only, info_testing_cv_only_with_fs, info_testing_cv_and_cli_nofs, info_testing_cv_only_modeld),
       col=c(color_list[6], color_list[4],color_list[1],color_list[3]), lty=1,cex=1.4, text.font=2.2, box.lty=2)

title('Testing ROCs of with Radiomics and Clinical Features',cex.main = 1.7, line = 2.2,xlab = "Specificity",ylab = "Sensitivity", cex.lab = 1.8)
#
###############################################################

######################COLLECT FPR TPR and Odds Ratio#######################
#########
fpr_tpr_test_cli_only= matrix(NA,length(stat_testing_cli_only$sensitivities),3)
sen_test_cli_only= stat_testing_cli_only$sensitivities
spec_test_cli_only= stat_testing_cli_only$specificities
for (i in (1:length(sen_test_cli_only))){
  odds_ratio_now = (spec_test_cli_only[[i]]*sen_test_cli_only[[i]])/((1-spec_test_cli_only[[i]])*(1-sen_test_cli_only[[i]]))
  fpr_tpr_test_cli_only[i,] = c(1-spec_test_cli_only[[i]],sen_test_cli_only[[i]],odds_ratio_now)
}
write.csv(matrix(fpr_tpr_test_cli_only, nrow=length(sen_test_cli_only)), file =paste(csv_saved_path,"fpr_tpr_oddsratio_test_cli_only.csv",sep=""), row.names=FALSE, col.names=FALSE)

#########
fpr_tpr_test_cv_and_cli_nofs= matrix(NA,length(stat_testing_cv_and_cli_nofs$sensitivities),3)
sen_test_cv_and_cli_nofs= stat_testing_cv_and_cli_nofs$sensitivities
spec_test_cv_and_cli_nofs= stat_testing_cv_and_cli_nofs$specificities
for (i in (1:length(sen_test_cv_and_cli_nofs))){
  odds_ratio_now = (spec_test_cv_and_cli_nofs[[i]]*sen_test_cv_and_cli_nofs[[i]])/((1-spec_test_cv_and_cli_nofs[[i]])*(1-sen_test_cv_and_cli_nofs[[i]]))
  fpr_tpr_test_cv_and_cli_nofs[i,] = c(1-spec_test_cv_and_cli_nofs[[i]],sen_test_cv_and_cli_nofs[[i]],odds_ratio_now)
}
#########
write.csv(matrix(fpr_tpr_test_cv_and_cli_nofs, nrow=length(sen_test_cv_and_cli_nofs)), file =paste(csv_saved_path,"fpr_tpr_oddsratio_test_cv_and_cli_nofs.csv",sep=""), row.names=FALSE, col.names=FALSE)
fpr_tpr_test_cv_only_with_fs= matrix(NA,length(stat_testing_cv_only_with_fs$sensitivities),3)
sen_test_cv_only_with_fs= stat_testing_cv_only_with_fs$sensitivities
spec_test_cv_only_with_fs= stat_testing_cv_only_with_fs$specificities
for (i in (1:length(sen_test_cv_only_with_fs))){
  odds_ratio_now = (spec_test_cv_only_with_fs[[i]]*sen_test_cv_only_with_fs[[i]])/((1-spec_test_cv_only_with_fs[[i]])*(1-sen_test_cv_only_with_fs[[i]]))
  fpr_tpr_test_cv_only_with_fs[i,] = c(1-spec_test_cv_only_with_fs[[i]],sen_test_cv_only_with_fs[[i]],odds_ratio_now)
}
write.csv(matrix(fpr_tpr_test_cv_only_with_fs, nrow=length(sen_test_cv_only_with_fs)), file =paste(csv_saved_path,"fpr_tpr_oddsratio_test_cv_only_with_fs.csv",sep=""), row.names=FALSE, col.names=FALSE)
#########
fpr_tpr_test_cv_only_modeld= matrix(NA,length(stat_testing_cv_only_modeld$sensitivities),3)
sen_test_cv_only_modeld= stat_testing_cv_only_modeld$sensitivities
spec_test_cv_only_modeld= stat_testing_cv_only_modeld$specificities
for (i in (1:length(sen_test_cv_only_modeld))){
  odds_ratio_now = (spec_test_cv_only_modeld[[i]]*sen_test_cv_only_modeld[[i]])/((1-spec_test_cv_only_modeld[[i]])*(1-sen_test_cv_only_modeld[[i]]))
  fpr_tpr_test_cv_only_modeld[i,] = c(1-spec_test_cv_only_modeld[[i]],sen_test_cv_only_modeld[[i]],odds_ratio_now)
}
write.csv(matrix(fpr_tpr_test_cv_only_modeld, nrow=length(sen_test_cv_only_modeld)), file =paste(csv_saved_path,"fpr_tpr_oddsratio_test_cv_only_modeld.csv",sep=""), row.names=FALSE, col.names=FALSE)

###############################################################
pr_testing_cv_only_nofs = pr.curve(scores.class0 = pred_testing_cv_only_nofs[which(labels_overall==1)],scores.class1 = pred_testing_cv_only_nofs[which(labels_overall==0)],curve = T)
pr_testing_cv_and_cli_nofs = pr.curve(scores.class0 = pred_testing_cv_and_cli_nofs[which(labels_overall==1)],scores.class1 = pred_testing_cv_and_cli_nofs[which(labels_overall==0)],curve = T)
pr_testing_cv_only_with_fs = pr.curve(scores.class0 = pred_testing_cv_only_with_fs[which(labels_overall==1)],scores.class1 = pred_testing_cv_only_with_fs[which(labels_overall==0)],curve = T)
pr_testing_cv_only_modeld = pr.curve(scores.class0 = pred_testing_cv_only_modeld[which(labels_overall==1)],scores.class1 = pred_testing_cv_only_modeld[which(labels_overall==0)],curve = T)
pr_testing_cli_only = pr.curve(scores.class0 = pred_testing_cli_only[which(labels_overall==1)],scores.class1 = pred_testing_cli_only[which(labels_overall==0)],curve = T)

###
# pr_info_testing_cv_only_nofs = paste("DCIS-R PRAUC: ",round(pr_testing_cv_only_nofs$auc.integral,3),sep = "")
pr_info_testing_cv_and_cli_nofs = paste("DCIS-RC PRAUC: ",round(pr_testing_cv_and_cli_nofs$auc.integral,3),sep = "")
pr_info_testing_cv_only_with_fs = paste("DCIS-Rs PRAUC: ",round(pr_testing_cv_only_with_fs$auc.integral,3),sep = "")
pr_info_testing_cv_only_modeld = paste("ADHIDC-R PRAUC: ",round(pr_testing_cv_only_modeld$auc.integral,3),sep = "")
pr_info_testing_cli_only = paste("DCIS-C PRAUC: ",round(pr_testing_cli_only$auc.integral,3),sep = "")

dev.off()
plot(seq(from=0.1, to  = 1, by = 0.01),seq(from=1.0, to  = 0.1, by = -0.01),type="l",lty = 2,col = "white",ann=FALSE,xaxt='n',yaxt='n')
par(new=TRUE)
plot(pr_testing_cli_only$curve[,1],pr_testing_cli_only$curve[,2], col = color_list[6], lwd = 4, lty = 1, type = "l",ann=FALSE,xaxt='n',yaxt='n')
par(new=TRUE)
plot(pr_testing_cv_only_with_fs$curve[,1],pr_testing_cv_only_with_fs$curve[,2], col = color_list[4], lwd = 4, lty = 1, type = "l",ann=FALSE,xaxt='n',yaxt='n')
par(new=TRUE)
plot(pr_testing_cv_and_cli_nofs$curve[,1],pr_testing_cv_and_cli_nofs$curve[,2], col = color_list[3], lwd = 4, lty = 1, type = "l",ann=FALSE,xaxt='n',yaxt='n')
par(new=TRUE)
plot(pr_testing_cv_only_modeld$curve[,1],pr_testing_cv_only_modeld$curve[,2], col = color_list[1], lwd = 4, lty = 1, type = "l",ann=FALSE,xaxt='n',yaxt='n')
par(new=TRUE)
points(x = 0.229166667,y = 0.239130435,pch = 18,col = "black",cex =  3)

axis(side=1,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 2.4)
axis(side=2,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 2.4)

legend(0.45, 1.02, legend=c("Clinical Rules' Operating Point: (23%, 24%)",pr_info_testing_cli_only,pr_info_testing_cv_only_with_fs, pr_info_testing_cv_and_cli_nofs, pr_info_testing_cv_only_modeld),
       col=c("black", color_list[6],color_list[4],color_list[3],color_list[1]), lty=c(NA,1,1,1,1), pch=c(18,NA,NA,NA,NA),cex=1.2, text.font=2.2, box.lty=2)

legend(0.61, 1, legend=c(pr_info_testing_cli_only,pr_info_testing_cv_only_with_fs, pr_info_testing_cv_and_cli_nofs, pr_info_testing_cv_only_modeld),
       col=c(color_list[6],color_list[4],color_list[1],color_list[3]), lty=1, cex=1.4, text.font=2.2, box.lty=2)

title('Models Prediction PR-Curves on Test Set',cex.main = 1.6, line = 2.2,xlab = "Recall",ylab = "Precision", cex.lab = 1.8)

plot(perf_testing_cli_only, col = color_list[6], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
plot(perf_testing_cv_only_with_fs, col = color_list[4], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
plot(perf_testing_cv_and_cli_nofs, col = color_list[3], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
plot(perf_testing_cv_only_modeld, col = color_list[1], lty = 1,lwd = 4,add = TRUE,xaxt='n',yaxt='n')
points(x = 0.14,y = 0.23,pch = 18,col = "black",cex =  3) # out of 300 cases, 2 cases don't have ER or PR or NG info

axis(side=1,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("1.0","0.9","0.8","0.7","0.6","0.5","0.4","0.3","0.2","0.1","0.0"),font.axis = 3)
axis(side=2,at=c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),labels=c("0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"),font.axis = 3)

