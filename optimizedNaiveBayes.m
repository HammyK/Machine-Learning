
datafile = readtable('bank_data.xlsx');
datafile.job = grp2idx(categorical(datafile.job));
datafile.marital = grp2idx(categorical(datafile.marital));
datafile.education = grp2idx(categorical(datafile.education));
datafile.default = grp2idx(categorical(datafile.default));
datafile.housing = grp2idx(categorical(datafile.housing));
datafile.loan = grp2idx(categorical(datafile.loan));
datafile.contact = grp2idx(categorical(datafile.contact));
datafile.month = grp2idx(categorical(datafile.month));
datafile.poutcome = grp2idx(categorical(datafile.poutcome));
datafile = datafile(:,[1 6 10 12 13 14 15 2 3 4 5 7 8 9 11 16 17]);
partition = cvpartition(datafile.y,'holdout',0.35);
trainset = datafile(training(partition),:);
testset = datafile(test(partition),:);
Resp = strcmp(trainset.y,"yes");
testset.y = strcmp(testset.y,"yes");
trainset.y = strcmp(trainset.y,"yes");

%Naive Bayes Model
%Calculation of losses on training and test dataset
NaiveB = fitcnb(trainset,'y');
NBLossTST = loss(NaiveB,testset);

%Calculating AUC Values and Scores
[NBXT,NBYT,NBTT,NBAUCT,NBPredictT] = fnAucRocValue(NaiveB,testset,false);

%Formulas to calculate accuracy,misclassification rate, precision & recall
[AccuracyNBTST,MisclassifNBTST,PrecisionNBTST,RecallNBTST] = fnAcc(NBPredictT,testset.y);

%---- Step ---- Improving predective models -Optimizing the model --------

%Model Optimisation/Tuning

%We tweak the model by changing some of its properties. We calculate the loss 
%and compare between default and customised naive bayes %models. This is 
%completed by using Multivariate Multinomial Distribution for categorical 
%columns and Kernel Smoothing Density estimate for numerical columns 
dist = [repmat("kernel",1,7),repmat("mvmn",1,9)];
NBCustMod = fitcnb(trainset,'y','DistributionNames',dist);
NBCustLossTST = loss(NBCustMod,testset);

%Calculating AUC Values and Scores
[NBCustomXT,NBCustomYT,NBCustomTT,NBCustomAUCT,NBCustomPredictT] = fnAucRocValue(NBCustMod,testset,false);

%Calculating model accuracy,misclassification rate, precision and recall
[AccuracyNBCustomXT_TST,MisclassifNBCustomXT_TST,PrecisionNBCustomXT_TST,RecallNBCustomXT_TST] = fnAcc(NBCustomPredictT,testset.y); 

%Defining Prior
Prior = [0.92 0.08];
Mdl = fitcnb(trainset,'y','Prior',Prior);

%We create a series of dummy predictors for categorical data in our training dataset using
%sequential feature selection. Using sequential feature selection with a nb model to determine 
%which variables to retain.
Stoploss = @(XT,yT,Xt,yt)loss(fitcnb(XT,yT),Xt,yt);
Seqpred = sequentialfs(Stoploss,trainset{:,1:end-1},trainset.y,'options',statset('Display','iter'));

%CV Naive Bayes model run with feature selected predictors and calculating
%model loss
FeatSelNBMD1 = fitcnb(trainset(:,[Seqpred true]),'y');
FeatSelNBMD1Loss= loss(FeatSelNBMD1,testset);

%HP Optimisation
NBHPMD1 = fitcnb(trainset(:,[Seqpred true]),'y','DistributionNames',dist,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
NBHPMD1LOSS = loss(NBHPMD1,testset);

%Calculating AUC Values and Scores
[NBHPX,NBHPY,NBHPT,NBHPAUC,NBHPPREDICT] = fnAucRocValue(NBHPMD1,trainset,false);
[NBHPXT,NBHPYT,NBHPTT,NBHPAUCT,NBHPPREDICTT] = fnAucRocValue(NBHPMD1,testset,false);

%Calculating model accuracy,misclassification rate, precision and recall
[AccuracyHP,MisclassifHP,PrecisionHP,RecallHP] = fnAcc(NBHPPREDICT,trainset.y); 
[AccuracyHP_TST,MisclassifHP_TST,PrecisionHP_TST,RecallHP_TST] = fnAcc(NBHPPREDICTT,testset.y);

%Here we create a default Naive Bayes Binary Model Classifier
t = templateNaiveBayes('DistributionNames',dist);
tempNBMD1 = fitcecoc(trainset,'y','Learners',t);
tempNBMD1Loss = loss(tempNBMD1,testset);

%Calculating AUC Values and Scores
[NBTempX,NBTempY,NBTempT,NBTempAUC,NBTempPredict] = fnAucRocValue(tempNBMD1,trainset,false);
[AccuracyNBTEMP,MisclassifNBTEMP,PrecisionNBTEMP,NBTEMPPredictT] = fnAucRocValue(tempNBMD1,testset,false);

%Calculating model accuracy,misclassification rate, precision and recall
[AccuracyNBTEMPTST,MisclassifNBTEMPTST,PrecisionNBTEMPTST,RecallNBTEMPTST] = fnAcc(NBTEMPPredictT,testset.y);

%Comparison of Loss across Naive Bayes models by calculating Accuracy/ROC
%Graph to visualise all model errors/loss function
LossModels = categorical(["Simple NB","Kernel MVNM NB","HyperParameter NB","Sequential Selection"]);
NBModelErrors = [NBLossTST,NBCustLossTST,NBHPMD1LOSS,FeatSelNBMD1Loss];
bar(LossModels,NBModelErrors)
ylabel("NB Model Losses");
title("Loss for all NB Models");

%ROC Plots
figure('Name','ROC Comparison Curves')
plot(NBHPX,NBHPY)
hold on
plot(NBHPXT,NBHPYT)
legend('HyperParameter NB_Train',"HyperParameter NB_Test")
xlabel('False positive') ; ylabel('True positive');
title('ROC Curves for Optimised Naive Bayes Model')
hold off

%accuracy,misclassification rate, precision and recall for all models
AccuracyValNB = [AccuracyNBTST,AccuracyNBCustomXT_TST,AccuracyHP_TST,AccuracyNBTEMPTST];
figure('Name','Accuracy Vals')
bar(LossModels,AccuracyValNB)
ylabel("Accuracy");
title("Accuracy Values of NB Models");

RecallValNB = [RecallNBTST,RecallNBCustomXT_TST,RecallHP_TST,RecallNBTEMPTST];
figure('Name','Recall Vals')
bar(LossModels,RecallValNB)
ylabel("Recall Values");
title("Recall Values of NB Models");

MisclassifcationValNB = [MisclassifNBTST,MisclassifNBCustomXT_TST,MisclassifHP_TST,MisclassifNBTEMPTST];
figure('Name','MisclassificationVals')
bar(LossModels,MisclassifcationValNB)
ylabel("Misclassified Loss");
title("Misclassified Loss of NB Models");

PrecisionValNB = [PrecisionNBTST,PrecisionNBCustomXT_TST,PrecisionHP_TST,PrecisionNBTEMPTST];
figure('Name','Precision Vals')
bar(LossModels,PrecisionValNB)
ylabel("Precision");
title("Precision Values of NB Models");

%Calculating model accuracy,misclassification rate, precision and recall
figure('Name','NB Model Confusion chart Test data');
confusionchart(testset.y,NBHPPREDICTT)
title("NB Model Confusion chart Test data");

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
