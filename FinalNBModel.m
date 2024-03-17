%Loading Dataset
%Column sort numerical to categorical
%Label Encoding (If Trainset is 'Yes' then 1, else 0 'No)
%Data Partitioning at 35%
dataset = readtable('bank_full.xlsx');
dataset.job = grp2idx(categorical(dataset.job));
dataset.marital = grp2idx(categorical(dataset.marital));
dataset.education = grp2idx(categorical(dataset.education));
dataset.default = grp2idx(categorical(dataset.default));
dataset.housing = grp2idx(categorical(dataset.housing));
dataset.loan = grp2idx(categorical(dataset.loan));
dataset.contact = grp2idx(categorical(dataset.contact));
dataset.month = grp2idx(categorical(dataset.month));
dataset.poutcome = grp2idx(categorical(dataset.poutcome));
dataset.y = strcmp(dataset.y,"yes");
partition = cvpartition(dataset.y,'holdout',0.35);
trainset = dataset(training(partition),:);
testset = dataset(test(partition),:);

%Naive Bayes Model
%We calculate the loss and compare between default and customised naive bayes
%models. This is completed by using Multivariate Multinomial Distribution for
%categorical columns and Kernel Smoothing Density estimate for numerical columns 
dist = [repmat("kernel",1,7),repmat("mvmn",1,9)];

%Feature selection to only those predictor columns which are relevant for 
%naive bayes model. 
%sequentialfs Sequential feature selection using custom criterion stoploss.
%Loss Function
Stoploss = @(XT,yT,Xt,yt)loss(fitcnb(XT,yT),Xt,yt);
Seqpred = sequentialfs(Stoploss,trainset{:,1:end-1},trainset.y,'options',statset('Display','iter'));

%Train and dataset are standardised for feature selected parameters
trainset.age = zscore(trainset.age);
trainset.job = zscore(trainset.job);
trainset.marital = zscore(trainset.marital);
trainset.contact = zscore(trainset.contact);
trainset.duration = zscore(trainset.duration);
testset.age = zscore(testset.age);
testset.job = zscore(testset.job);
testset.marital = zscore(testset.marital);
testset.contact = zscore(testset.contact);
testset.duration = zscore(testset.duration);

%Find hyperparameters that minimize five-fold cross-validation loss by 
%using automatic hyperparameter optimization. This is set to 'auto' which then 
% optimises using name and width. For reproducibility,the random seed is 
% set and the 'expected-improvement-plus' acquisition function is used. 
% Computational speed is increased by using parallel pooling. Tic Toc used
% to find the calculation time.

rng('default');

tic;
naiveHPMD1 = fitcnb(trainset(:,[Seqpred true]),'y','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','UseParallel',true));
toc;

%Calculation of losses on training and test dataset
naiveHPMD1Loss_TRN = resubLoss(naiveHPMD1);
naiveHPMD1Loss_TST = loss(naiveHPMD1,testset);

%Calculation for Threshold,AUC,ROC and label predictions
[nbhpX,nbhpY,nbhpT,nbhpAUC,nbhpPredict] = fnAucRocValue(naiveHPMD1,trainset);
[nbhpXT,nbhpYT,nbhpTT,nbhpAUCT,nbhpPredictT] = fnAucRocValue(naiveHPMD1,testset); 

%Calculating model accuracy,misclassification rate, precision, recall &
%F-score
[Accuracynnbhp,Misclassifnbhp,Precisionnbhp,Recallnbhp,Fscorenbhp] = fnAcc(nbhpPredict,trainset.y); 
[Accuracynnbhp_TST,Misclassifnbhp_TST,Precisionnbhp_TST,Recallnbhp_TST,Fscorenbhp_TST] = fnAcc(nbhpPredictT,testset.y); 

%ROC Plot
figure('Name','ROC Comparisons')
plot(nbhpX,nbhpY) 
hold on
plot(nbhpXT,nbhpYT)
legend('Naive Bayes HP Trainset',"Naive Bayes HP Testset")
xlabel('False positive rate') ; ylabel('True positive rate');
title('ROC Curves for Optimised Naive Bayes')
hold off

%Calculating model accuracy,misclassification rate, precision, recall &
%F-score for Test data
PerformanceMetrics = categorical(["Accuracy","Precision","Recall","Fscore","Misclassified Rate"]);
Values = [Accuracynnbhp_TST,Precisionnbhp_TST,Recallnbhp_TST,Fscorenbhp_TST,Misclassifnbhp_TST];
figure('Name','Performance Metrics Plot')
bar(PerformanceMetrics,Values)
title("Performance Parameters Plot for Naive bayes");

%Confusion matrix on the final optimised model with Test dataset
figure('Name','Naive Bayes Confusion Matrix')
confusionchart(testset.y,nbhpPredictT)
title("Performance Metrics Plot for Naive Bayes");

%Calculating model accuracy,misclassification rate, precision, recall &
%F-score for Test data
function [Accuracy,Misclassif,Precision,Recall,Fscore] = fnAcc(Predictions,RealLabels)
isCorrect = Predictions==RealLabels; %Correct Label Prediction
isWrong = Predictions~=RealLabels; %Incorrect Label Prediction
Accuracy = sum(isCorrect)/numel(Predictions);
Misclassif = sum(isWrong)/numel(Predictions); %Misclassification Rate

%Confusion Matrix
[cmatrix,order] = confusionmat(RealLabels,Predictions);
for i =1:size(cmatrix,1)
    Precision(i)=cmatrix(i,i)/sum(cmatrix(i,:));
end
for i =1:size(cmatrix,1)
    Recall(i)=cmatrix(i,i)/sum(cmatrix(:,i));
end
Precision=sum(Precision)/size(cmatrix,1);
Recall=sum(Recall)/size(cmatrix,1);
Fscore = (2*Recall*Precision/(Recall+Precision));
end

%Formulas to calculate AUC/Threshold and Predicted Values/Scores
function [X,Y,T,Auc,Label] = fnAucRocValue(mdl,data)
[Label,Score] = predict(mdl,data);
[X,Y,T,Auc] = perfcurve(data.y,Score(:,mdl.ClassNames),'true');
end