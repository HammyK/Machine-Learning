%Loading Dataset
%Column sort numerical to categorical
%Label Encoding (If Trainset is 'Yes' then 1, else 0 'No)
%Data Partitioning at 35%
datafile = readtable('bank_data.xlsx');
datafile = datafile(:,[1 6 10 12 13 14 15 2 3 4 5 7 8 9 11 16 17]);
partition = cvpartition(datafile.y,'holdout',0.35);
trainset = datafile(training(partition),:);
testset = datafile(test(partition),:);
Resp = strcmp(trainset.y,"yes");
testset.y = strcmp(testset.y,"yes");
trainset.y = strcmp(trainset.y,"yes");

%Decision Tree Model
%Simple Tree Model
%Calculation of losses on training and test dataset
DTPredict= fitctree(trainset,'y','SplitCriterion', 'gdi','MaxNumSplits', 100);
TreelossHPOpt = resubLoss(DTPredict);
Treeloss_test = loss(DTPredict,testset);
disp('Training Error: ' + TreelossHPOpt);
disp('Testing Error: ' + Treeloss_test);

%Calculating AUC/Threshold Values and Scores
[Label_TRN,Score_TRN] = predict(DTPredict,trainset); % Train dataset
[Label_TST,Score_TST] = predict(DTPredict,testset); %Test dataset
[XT,YT,TT,AUCTh] = perfcurve(Resp,Score_TRN(:,DTPredict.ClassNames),'true');

%Model Optimisation/Tuning
%Pruning our Tree
PrunedMD1 = prune(DTPredict,'Level',12); 

%Calculation of losses on training and test dataset
Treeloss_Prunetrain = resubLoss(PrunedMD1);
Treeloss_Prunetest = loss(PrunedMD1,testset);
disp('Pruned Training Error : ' + Treeloss_Prunetrain);
disp('Pruned Test Error : ' + Treeloss_Prunetest);

%Calculating AUC/Threshold Values and Scores on Pruned
[~,PrunedScoreT] = resubPredict(PrunedMD1);
[XTPruned,YTPruned,TTPruned,AUCTPruned] = perfcurve(Resp,PrunedScoreT(:,PrunedMD1.ClassNames),'true');

% Graph for Pruned Tree
view(PrunedMD1,'mode','graph');
view(DTPredict,'mode','graph');
 
%Model Tuning
%CV with folds for least classification loss\k-fold classification error
LeastLoss_KFold = [] ;
for n = 2:10
    CVTreeMD1CK= fitctree(trainset,'y','KFold',n);
    Loss_KFold =  kfoldLoss(CVTreeMD1CK);
    if isempty(LeastLoss_KFold)
       LeastLoss_KFold = Loss_KFold;
       Folds = n;
       CVTreeMD1 = CVTreeMD1CK;
    elseif (Loss_KFold < LeastLoss_KFold)
        LeastLoss_KFold = Loss_KFold;
        Folds = n;
        CVTreeMD1 = CVTreeMD1CK;
    end    
end
disp(Folds +'Tree Error on Fold Decisions : ' + LeastLoss_KFold);

%Calculating AUC/Threshold Values and Scores
[~,scoreCVT] = kfoldPredict(CVTreeMD1);
[XTCV,YTCV,TTCV,AUCTCV] = perfcurve(Resp,scoreCVT(:,CVTreeMD1.ClassNames),'true');

%Tree Optimisation using Hyperparameters
rng('default') 
tallrng('default')
HPOptTreeMdl = fitctree(trainset,Resp,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Holdout',0.3,...
    'AcquisitionFunctionName','expected-improvement-plus'))

%Calculating k-Fold Loss 
TreelossHPOpt = resubLoss(HPOptTreeMdl);

%Calculating AUC/Threshold Values and Scores
[~,scoreHPOptTree] = resubPredict(HPOptTreeMdl);
[XTHP,YTHP,TTHP,AUCTHP] = perfcurve(Resp,scoreHPOptTree(:,HPOptTreeMdl.ClassNames),'true');

%Feature Transformation PCA (Principle Component Analysis)
%Array for numerical columns Loop
%1 for numerical and 0 for categorical columns
%Parting Categorical Columns
Index = 1;
for n = 1:16
    if(isnumeric(trainset{:,n})) 
         numCol(Index)= true; 
    else
         numCol(Index)=false;
    end     
         Index=Index+1;
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
predictors = [array2table(scrs(:,1:Comp_Retained)), trainset(:, ~numCol)];

%Tree created using features selected from PCA and model loss shown
TreeMD1PCA = fitctree(predictors(:,1:end-1),trainset.y,"KFold",Folds);
TreeMD1PCALoss = kfoldLoss(TreeMD1PCA);

%Calculating AUC/Threshold Values and Scores
[~,ScoreTPCA] = kfoldPredict(TreeMD1PCA);
[XTPCA,YTPCA,TTPCA,AUCTPCA] = perfcurve(Resp,ScoreTPCA(:,TreeMD1PCA.ClassNames),'true');

%Feature Selection using MATLAB Function
Predictors = predictorImportance(DTPredict);
ColName = categorical(datafile.Properties.VariableNames(1:end-1));
bar(ColName,Predictors)
ylabel("Predictor Importance");
xlabel("Column Names");
title("Predictor Importance on Y");

RelevantPredictors = Predictors >0.00006;
RelPredict_train = trainset(:,[RelevantPredictors true]);

%CV Tree model with relevant predictor columns created using folds and
%calculating loss
RelPredTreeMD1= fitctree(RelPredict_train,trainset.y',"KFold",Folds);
RelPredTreeMD1_Loss = kfoldLoss(RelPredTreeMD1);

%Calculating AUC/Threshold Values and Scores
[~,ScoreRPTMD1] = kfoldPredict(RelPredTreeMD1);
[XTRPT,YTRPT,TTRPT,AUCTRPT] = perfcurve(Resp,ScoreRPTMD1(:,RelPredTreeMD1.ClassNames),'true');

%Bagged trees created using tree learners in ensemble for training dataset
%with output "y" and calculating K-Fold Loss
EnsemTreeMD1 = fitcensemble(trainset,"y","Learners","tree","Method","Bag","NumLearningCycles",30,...
"KFold",Folds);
EnsemKFoldloss = kfoldLoss(EnsemTreeMD1);

%Calculating AUC/Threshold Values and Scores
[~,EnsemScoreT] = kfoldPredict(EnsemTreeMD1);
[XTEnsem,YTEnsem,TTEnsem,AUCTEnsem] = perfcurve(Resp,EnsemScoreT(:,EnsemTreeMD1.ClassNames),'true');

%Comparison of Loss across Decision Tree models by calculating Accuracy/ROC
%Graph to visualise all tree model errors/loss function
LossModels = categorical(["Simple Tree","Pruned Tree", "CV Tree","PCA CV Tree","Ensemble CV Tree"]);
ModelErrors = [Treeloss_test,Treeloss_Prunetest,LeastLoss_KFold,TreeMD1PCALoss,EnsemKFoldloss];
bar(LossModels,ModelErrors)
ylabel("Misclassified Loss");
title("Misclassified Errors for all Models");

%ROC Plots
figure('Name','ROC Comparison Curves')
plot(XT,YT)
hold on
plot(XTPruned,YTPruned)
plot(XTCV,YTCV)
plot(XTPCA,YTPCA)
plot(XTEnsem,YTEnsem)
legend("Simple Tree","Pruned Tree","CV Tree","PCA CV Tree","Ensemble CV Tree")
xlabel('False positive') ; ylabel('True positive');
title('ROC Curves for Simple Tree, Pruned Tree ,CV Tree ,PCA CV Tree, Ensemble CV Tree')
hold off

%Formulas to calculate accuracy,misclassification rate, precision & recall
function [Accuracy,Misclassif,Precision,Recall] = fnAcc(Predictions,RealLabels)
CorrLabPred = Predictions==RealLabels;
InCorrLabPred = Predictions~=RealLabels;
Accuracy = sum(CorrLabPred)/numel(Predictions);
Misclassif = sum(InCorrLabPred)/numel(Predictions);
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
function [X,Y,T,Auc] = fnAucRocValue(mdl,Resp)
[Label,Score] = kfoldPredict(mdl);
[X,Y,T,Auc] = perfcurve(Resp,Score(:,mdl.ClassNames),'true');
end