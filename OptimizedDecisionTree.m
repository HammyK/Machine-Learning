%Loading Dataset
%Column sort numerical to categorical
%Label Encoding (If Trainset is 'Yes' then 1, else 0 'No)
%Data Partitioning at 35%
datafile = readtable('bank_full.xlsx');
datafile = datafile(:,[1 6 10 12 13 14 15 2 3 4 5 7 8 9 11 16 17]);
datafile.y = strcmp(datafile.y,"yes");
partition = cvpartition(datafile.y,'holdout',0.35);
trainset = datafile(training(partition),:);
testset = datafile(test(partition),:);

%Decision Tree Model
%Simple Tree Model
%Calculation of losses on training and test dataset
DTPredict= fitctree(trainset,'y','SplitCriterion', 'gdi','MaxNumSplits', 100);
TreelossHPOpt = resubLoss(DTPredict);
Treeloss_test = loss(DTPredict,testset);

%ROC Curve/AUC Value - Calculation
[XTST,YTST,TTST,AUCTST,PredictTRN] = fnAucRocValue(DTPredict,trainset,false);
[XT,YT,TT,AUCT,PredictTST] = fnAucRocValue(DTPredict,testset,false);

%Calculation of Accuracy,Recall, Precision and Misclassification
[AccuracyT,MisclassifT,PrecisionT,RecallT] = fnAcc(PredictTRN,trainset.y); 
[AccuracyTST,MisclassifT,PrecisionTST,RecallTST] = fnAcc(PredictTST,testset.y); 

%Pruning
PrunedMD1 = prune(DTPredict,'Level',12); 

%Calculation of losses on training and test dataset
Treeloss_Prunetrain = resubLoss(PrunedMD1);
Treeloss_Prunetest = loss(PrunedMD1,testset);

%AUC Values Calculation
[XTPruned,YTPruned,TtPruned,AucTPruned,prdctPrunedT] = fnAucRocValue(PrunedMD1,trainset,false);
[XTSTPruned,YTSTPruned,TTSTPruned,AUCTSTPruned,PredictPrunedTST] = fnAucRocValue(PrunedMD1,testset,false);

%Calculation of Accuracy,Recall, Precision and Misclassification
[AccuracyPrunedT,MisclassifPrunedT,PrecisionPrunedT,RecallPrunedT] = fnAcc(prdctPrunedT,trainset.y); 
[AccuracyPrunedTST,MisclassifPrunedTST,PrecisionPrunedTST,RecallPrunedTST] = fnAcc(PredictPrunedTST,testset.y); 

%Graph for Pruned Trees
view(PrunedMD1,'mode','graph');
view(DTPredict,'mode','graph');
 
%Feature Selection using MATLAB Function
Predictors = predictorImportance(DTPredict);
ColNames = categorical(datafile.Properties.VariableNames(1:end-1));
bar(ColNames,Predictors)
ylabel("Predictor Importance");
xlabel("Column Names");
title("Predictor Importance on Y");

RelevantPredictors = Predictors >0.00006;

bankPartTrain = trainset(:,[RelevantPredictors true]);
bankPartTest = testset(:,[RelevantPredictors true]);

%CV Tree model with relevant predictor columns created using folds and
%calculating loss
RelPredTreeMD1= fitctree(bankPartTrain,'y');
RelPredTreeMD1_TRNLoss = resubLoss(RelPredTreeMD1);
RelPredTreeMD1_TSTLoss = loss(RelPredTreeMD1,bankPartTest);

%Calculating AUC/Threshold Values and Scores
[XTRPT,YTRPT,TTRPT,AUCRPT,PredictRPT]   = fnAucRocValue(RelPredTreeMD1,bankPartTrain,false);
[XTSTRPT,YTSTRPT,TTSTRPT,AUCTSTRPT,PredictTSTRPT]   = fnAucRocValue(RelPredTreeMD1,bankPartTest,false);

%Calculation of Accuracy,Recall, Precision and Misclassification
[AccuracyRPTT,MisclassifRPTT,PrecisionRPTT,RecallRPTT] = fnAcc(PredictRPT,bankPartTrain.y);
[AccuracyRPTTST,MisclassifRPTTST,PrecisionRPTTST,RecallRPTTST] = fnAcc(PredictTSTRPT,bankPartTest.y);

%Tree Optimisation using Hyperparameters
rng('default') 
tallrng('default')

TT = templateTree();
MD1 = fitcensemble(bankPartTrain,'y','OptimizeHyperparameters','auto','Learners',TT, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

%Calculating Loss 
Treeloss_train = resubLoss(MD1);
Treeloss_test = loss(MD1,bankPartTest);

[XTHYP,YTHYP,TTHYP,AUCTHYP,PredictHYP] = fnAucRocValue(MD1,bankPartTrain,false);
[XTSTHYP,YTSTHYP,TTSTHYP,AUCTSTHYP,PredictTSTHYP] = fnAucRocValue(MD1,bankPartTest,false);

%Calculation of Accuracy,Recall, Precision and Misclassification
[AccuracyHYPT,MisclassifHYPT,PrecisionHYPT,RecallHYPT] = fnAcc(PredictHYP,bankPartTrain.y); 
[AccuracyHYPTST,MisclassifHYPTST,PrecisionHYPTST,RecallHYPTST] = fnAcc(PredictTSTHYP,bankPartTest.y); 

%Comparison of Loss across Decision Tree models by calculating Accuracy/ROC
LossModels = categorical(["Simple Tree","Pruned Tree","Feature Selected Tree","Optimised Feature Selected HP Tree"]);

figure('Name','ROC Comparison Curves')
plot(XTSTHYP,YTSTHYP)
hold on
plot(XTHYP,YTHYP)
legend("ROC for Test set","ROC for Train set")
xlabel('False Positive') ; ylabel('True Positive');
title('ROC Curves for Decision Trees')
hold off

%Calculation of Accuracy,Recall, Precision and Misclassification on DT
%Models
AccuracyVal = [AccuracyTST,AccuracyPrunedTST,AccuracyRPTTST,AccuracyHYPTST];
figure('Name','Accuracy Values')
bar(LossModels,AccuracyVal)
ylabel("Accuracy");
title("Decision Tree Models Accuracy on Test Dataset");

RecallVal = [RecallTST,RecallPrunedTST,RecallRPTTST,RecallHYPTST];
figure('Name','Recall Values')
bar(LossModels,RecallVal)
ylabel("Recall");
title("Decision Tree Models Recall on Test Dataset");

MisclassifRate = [MisclassifT,MisclassifPrunedTST,MisclassifRPTTST,MisclassifHYPTST];
figure('Name','Misclassification Rate')
bar(LossModels,MisclassifRate)
ylabel("Misclassified Loss");
title("Decision Tree Models Misclassification on Test Dataset");

PrecisionVal = [PrecisionTST,PrecisionPrunedTST,PrecisionRPTTST,PrecisionHYPTST];
figure('Name','Precision Values')
bar(LossModels,PrecisionVal)
ylabel("Precision");
title("Decision Tree Models Precision on Test Dataset");

%Optimised Model Confusion Matrix
figure('Name','Confusion Matrix for Decision Tree')
confusionchart(bankPartTest.y,PredictTSTHYP)

%Calculation of Accuracy,Recall, Precision and Misclassification
function [Accuracy,MisclassifRate,Precision,Recall] = fnAcc(Predictions,RealLabels)
CorrLabPred = Predictions==RealLabels;
InCorrLabPred = Predictions~=RealLabels;
Accuracy = sum(CorrLabPred)/numel(Predictions);
MisclassifRate = sum(InCorrLabPred)/numel(Predictions);
[cmatrixx,order] = confusionmat(RealLabels,Predictions);
for i =1:size(cmatrixx,1)
    Precision(i)=cmatrixx(i,i)/sum(cmatrixx(i,:));
end
for i =1:size(cmatrixx,1)
    Recall(i)=cmatrixx(i,i)/sum(cmatrixx(:,i));
end
Precision=sum(Precision)/size(cmatrixx,1);
Recall=sum(Recall)/size(cmatrixx,1);
end

%Calculating AUC/Threshold Values and Scores
function [X,Y,T,Auc,Label] = fnAucRocValue(mdl,data,cv)
if(cv==0)
    [Label,Score] = predict(mdl,data);
else
    [Label,Score] = kfoldPredict(mdl);
end    
[X,Y,T,Auc] = perfcurve(data.y,Score(:,mdl.ClassNames),'true');
end