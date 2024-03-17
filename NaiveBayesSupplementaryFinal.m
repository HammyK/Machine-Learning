%Loading Dataset
%Column sort numerical to categorical
%Label Encoding (If Trainset is 'Yes' then 1, else 0 'No)
%Data Partitioning at 35%
datafile = readtable('bank_data.xlsx');
partition = cvpartition(datafile.y,'holdout',0.35);
trainset = datafile(training(partition),:);
testset = datafile(test(partition),:);
Resp = strcmp(trainset.y,"yes");
testset.y = strcmp(testset.y,"yes");
trainset.y = strcmp(trainset.y,"yes");

%Naive Bayes Model
%Calculation of losses on training and test dataset
NaiveB = fitcnb(trainset,'y');
NBLossTRN = resubLoss(NaiveB);
NBLossTST = loss(NaiveB,testset);

%Calculating AUC Values and Scores 
[NBX,NBY,NBT,NBAUC,NBPredictTRN] = fnAucRocValue(NaiveB,trainset,false);
[NBXT,NBYT,NBTT,NBAUCT,NBPredictT] = fnAucRocValue(NaiveB,testset,false);

%Formulas to calculate accuracy,misclassification rate, precision & recall
[AccuracyNBT,MisclassifNBT,PrecisionNBT,RecallNBT] = fnAcc(NBPredictTRN,trainset.y); 
[AccuracyNBTST,MisclassifNBTTST,PrecisionNBTTST,RecallNBTTSTS] = fnAcc(NBPredictT,testset.y);

%Model Optimisation/Tuning

%We tweak the model by changing some of its properties. We calculate the loss 
%and compare between default and customised naive bayes %models. This is 
%completed by using Multivariate Multinomial Distribution for categorical 
%columns and Kernel Smoothing Density estimate for numerical columns 
 
dist = [repmat("kernel",1,1), repmat("mvmn",1,4), repmat("kernel",1,1) ,repmat("mvmn",1,3)...
    repmat("kernel",1,1), repmat("mvmn",1,1), repmat("kernel",1,4), repmat("mvmn",1,1)];
NBCustMod = fitcnb(trainset,'y','DistributionNames',dist);
NBCustLossTRN = resubLoss(NBCustMod);
NBCustLossTST = loss(NBCustMod,testset);
disp('Custom Model Loss Naive Bayes: ' + NBCustLossTST)

%Calculating AUC Values and Scores
[NBCustomX,NBCustomY,NBCustomT,NBCustomAUC,NBCustomPredict] = fnAucRocValue(NBCustMod,trainset,false);
[NBCustomXT,NBCustomYT,NBCustomTT,NBCustomAUCT,NBCustomPredictT] = fnAucRocValue(NBCustMod,testset,false);

%Calculating model accuracy,misclassification rate, precision and recall
[AccuracyNBCustomXT,MisclassifNBCustomXT,PrecisionNBCustomXT,RecallNBCustomXT] = fnAcc(NBCustomPredict,trainset.y); 
[AccuracyNBCustomXT_TST,MisclassifNBCustomXT_TST,PrecisionNBCustomXT_TST,RecallNBCustomXT_TST] = fnAcc(NBCustomPredictT,testset.y); 

%Model Tuning
%CV with folds for least classification loss\k-fold classification error.
%CVNBMD1 stores the least loss k-fold properties to our model.
NBLeastLoss_KFold = [] ;
for n = 2:10
    CVNBMD1= fitcnb(trainset,'y','KFold',n);
    Loss_KFold =  kfoldLoss(CVNBMD1);
    if isempty(NBLeastLoss_KFold)
       NBLeastLoss_KFold = Loss_KFold;
       folds = n;
       CVNBMD1= fitcnb(trainset,'y','KFold',n);
    elseif (Loss_KFold < NBLeastLoss_KFold)
        NBLeastLoss_KFold = Loss_KFold;
        folds = n;
        CVNBMD1= fitcnb(trainset,'y','KFold',n);
    end    
end
disp(folds +'Folds Error : ' + NBLeastLoss_KFold);

%Calculating AUC Values and Scores
[NBCVX,NBCVY,NBCVT,NBCVAUC,NBCVPredict] = fnAucRocValue(CVNBMD1,trainset,true);

%Calculating model accuracy,misclassification rate, precision and recall
[AccuracyNBCV,MisclassifNBCV,PrecisionNBCV,RecallNBCV] = fnAcc(NBCVPredict,trainset.y);

%%Feature Transformation PCA (Principle Component Analysis)
%Array for numerical columns Loop
%1 for numerical and 0 for categorical columns
%Parting Categorical Columns
index = 1;
for n = 1:16
    if(isnumeric(trainset{:,n}))
         numCol(index)= true; 
    else
         numCol(index)=false;
    end     
         index=index+1;
end  

categoricalCol = trainset(:, ~numCol);

%PCA on numerical predictors on train dataset. Categorical predictors had
%no PCA on them.
Dataset_TrainPCA = trainset(:,[numCol true]);        
[pcs,scrs,~,~,PExp] = pca(Dataset_TrainPCA{:,1:end-1});

%Retained Components explaining desired variance
ExpVar = 95/100;
Comp_Retained = find(cumsum(PExp)/sum(PExp) >= ExpVar, 1);
pcs = pcs(:,1:Comp_Retained);
 
% Graph - Explained Variance
pareto(PExp)
title("Explained Variance (PCA)");

%Numerical Column Names
%Finding how each predictor contributes to the principal component
Numerical_Name = Dataset_TrainPCA.Properties.VariableNames(1:end-1);
heatmap(abs(pcs),"YDisplayLabels",Numerical_Name)
ylabel("Principal Components");
title("Predictor Contribution to Principal Components");
Dummy_Predictors = [array2table(scrs(:,1:Comp_Retained)), trainset(:, ~numCol)];

%Tree created using features selected from PCA and model loss shown
NBMD1PCA = fitcnb(Dummy_Predictors(:,1:end-1),Resp,"KFold",folds);
NBMD1PCALoss = kfoldLoss(NBMD1PCA);

%Calculating AUC Values and Scores 
[NBPCAX,NBPCAY,NBPCAT,NBPCAAUC,NBPCAPREDICT] = fnAucRocValue(NBMD1PCA,trainset,true);

%Formulas to calculate accuracy,misclassification rate, precision & recall
[AccuracyNBPCA,MisclassifNBPCA,PrecisionNBPCA,RecallNBPCA] = fnAcc(NBPCAPREDICT,trainset.y); 

%HP Optimisation
NBHPMD1 = fitcnb(trainset,'y','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
NBHPMD1LOSS = loss(NBHPMD1,testset);

%Calculating AUC Values and Scores 
[NBHPX,NBHPY,NBHPT,NBHPAUC,NBHPPREDICT] = fnAucRocValue(NBHPMD1,trainset,false);
[NBHPXT,NBHPYT,NBHPTT,NBHPAUCT,NBHPPREDICTT] = fnAucRocValue(NBHPMD1,testset,false);

%Formulas to calculate accuracy,misclassification rate & precision
[AccuracyHP,MisclassifHP,PrecisionHP,RecallHP] = fnAcc(NBHPPREDICT,trainset.y); 
[AccuracyHP_TST,MisclassifHP_TST,PrecisionHP_TST,RecallHP_TST] = fnAcc(NBHPPREDICTT,testset.y);

%Here we create a default Naive Bayes Binary Model Classifier
tnb = templateNaiveBayes();
tempNBMD1 = fitcecoc(trainset,'y','CrossVal','on','Learners',tnb);
tempNBMD1Loss = kfoldLoss(tempNBMD1,'LossFun','ClassifErr');

%Calculating AUC Values and Scores
[NBTempX,NBTempY,NBTempT,NBTempAUC,NBTempPredict] = fnAucRocValue(tempNBMD1,trainset,true);

%Formulas to calculate accuracy,misclassification rate & precision
[AccuracyNBTEMP,MisclassifNBTEMP,PrecisionNBTEMP,RecallNBTEMP] = fnAcc(NBTempPredict,trainset.y); 

%We create a series of dummy predictors for categorical data in our training dataset using
%sequential feature selection.
 dummyJob = dummyvar(categorical(trainset.job));
 dummyMarital = dummyvar(categorical(trainset.marital));
 dummyEducation = dummyvar(categorical(trainset.education));
 dummyDefault = dummyvar(categorical(trainset.default));
 dummyHousing = dummyvar(categorical(trainset.housing));
 dummyLoan = dummyvar(categorical(trainset.loan));
 dummyContact = dummyvar(categorical(trainset.contact));
 dummyMonth = dummyvar(categorical(trainset.month));
 dummyPoutcome = dummyvar(categorical(trainset.poutcome));

%Matrix to show numerical predictors followed by dummy predictors .
Dummy_Predictors = [Dataset_TrainPCA{:,1:end-1} dummyJob dummyMarital dummyEducation dummyDefault dummyHousing dummyLoan dummyContact dummyMonth dummyPoutcome];

%Using sequential feature selection with a nb model to determine which variables to retain.
Stoploss = @(XT,yT,Xt,yt)loss(fitcnb(XT,yT),Xt,yt);
Seqpred = sequentialfs(Stoploss,Dummy_Predictors,trainset.y,'options',statset('Display','iter'));

%CV Naive Bayes model run with feature selected predictors and calculating
%model loss
FeatSelNBMD1 = fitcnb(Dummy_Predictors(:,Seqpred),Resp,"KFold",folds);
FeatSelNBMD1Loss= kfoldLoss(FeatSelNBMD1);

%Calculating AUC Values and Scores
[FeatSelNBX,FeatSelNBY,FeatSelNBT,FeatSelNBAUC,FeatSelNBPredict]  = fnAucRocValue(tempNBMD1,trainset,true);

%Formulas to calculate accuracy,misclassification rate & precision
[AccuracyFeatSelNB,MisclassifFeatSelNB,PrecisionFeatSelNB,RecallFeatSelNB] = fnAcc(FeatSelNBPredict,trainset.y);

%Comparison of Loss across Naive Bayes models by calculating Accuracy/ROC
%Graph to visualise all model errors/loss function
LossModels = categorical(["Simple NB","Kernel MVNM NB", "CV NB","PCA CV NB","HyperParameter NB","Learner NB","Feature Selected CV NB"]);
NBModelErrors = [NBLossTST,NBCustLossTST,NBLeastLoss_KFold,NBMD1PCALoss,NBHPMD1LOSS,tempNBMD1Loss,FeatSelNBMD1Loss];
bar(LossModels,NBModelErrors)
ylabel("Misclassified Loss");
title("Misclassified Errors for all Models");

%ROC Plots
figure('Name','ROC Comparison Curves')
plot(NBX,NBY)
hold on
plot(NBCustomX,NBCustomY)
plot(NBCVX,NBCVY)
plot(NBPCAX,NBPCAY)
plot(NBHPX,NBHPY)
plot(NBTempX,NBTempY)
plot(FeatSelNBX,FeatSelNBY)
legend('Simple NB','Kernel MVNM NB','CV NB','PCA CV NB','HyperParameter NB',"Learner NB","Feature Selected CV NB")
xlabel('False positive') ; ylabel('True positive');
title('ROC Curves for Simple NB,Kernel MVNM NB,CV NB,PCA CV NB,HyperParameter NB,Learner NB and Feature Selected CV NB')
hold off

%Comparing across accuracy,misclassification rate, precision & recall
AccuracyValNB = [AccuracyNBT,AccuracyNBCustomXT,AccuracyNBCV,AccuracyNBPCA,AccuracyHP,AccuracyNBTEMP,AccuracyFeatSelNB];
figure('Name','Accuracy Vals')
bar(LossModels,AccuracyValNB)
ylabel("Accuracy");
title("Accuracy Values of NB Models");

RecallValNB = [RecallNBT,RecallNBCustomXT,RecallNBCV,RecallNBPCA,RecallHP,RecallNBTEMP,RecallFeatSelNB];
figure('Name','Recall Vals')
bar(LossModels,RecallValNB)
ylabel("Recall");
title("Recall Values of NB Models");

MisclassifcationValNB = [MisclassifNBT,MisclassifNBCustomXT,MisclassifNBCV,MisclassifNBPCA,MisclassifHP,MisclassifNBTEMP,MisclassifFeatSelNB];
figure('Name','Misclassification Vals')
bar(LossModels,MisclassifcationValNB)
ylabel("Misclassified Loss");
title("Misclassified Loss of NB Models");

PrecisionValNB = [PrecisionNBT,PrecisionNBCustomXT,PrecisionNBCV,PrecisionNBPCA,PrecisionHP,PrecisionNBTEMP,PrecisionFeatSelNB];
figure('Name','Precision Vals')
bar(LossModels,PrecisionValNB)
ylabel("Precision");
title("Precision Values of NB Models");

%Formulas to calculate accuracy,misclassification rate, precision & recall
function [Accuracy,Misclassification,Precision,Recall] = fnAcc(Predictions,RealLabels)
CorrLabPred = Predictions==RealLabels;
InCorrLabPred = Predictions~=RealLabels;
Accuracy = sum(CorrLabPred)/numel(Predictions);
Misclassification = sum(InCorrLabPred)/numel(Predictions);
[cmatrix,order] = confusionmat(RealLabels,Predictions);
for i =1:size(cmatrix,1)
    Precision(i)=cmatrix(i,i)/sum(cmatrix(i,:));
end
for i =1:size(cmatrix,1)
    Recall(i)=cmatrix(i,i)/sum(cmatrix(:,i));
end
Precision=sum(Precision)/size(cmatrix,1);
Recall=sum(Recall)/size(cmatrix,1);
end

%Formulas to calculate AUC/Threshold and Predicted Values/Scores
function [X,Y,T,Auc,Label] = fnAucRocValue(mdl,data,cv)
if(cv==0)
    [Label,Score] = predict(mdl,data);
else
    [Label,Score] = kfoldPredict(mdl);
end    
[X,Y,T,Auc] = perfcurve(data.y,Score(:,mdl.ClassNames),'true');
end
