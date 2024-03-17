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
%Feature selection to only those predictor columns which are relevant for 
%decision tree model. Matlab built-in function is used to estimate 
%predictor importance for the tree and storing in function called 'Predictors'
%MaxNumSplits to control depth of tree.
%'DTPredict' model to forecast the relevance of each column/predictors using
%fitctree to obtain fitted binary classification decision tree based on
%input variables.
DTPredict= fitctree(trainset,'y','SplitCriterion', 'gdi','MaxNumSplits', 100);
Predictors = predictorImportance(DTPredict);

%Bar Chart for Predictor Importance
ColName = categorical(datafile.Properties.VariableNames(1:end-1));
bar(ColName,Predictors)
ylabel("Predictor Importance");
xlabel("Column Names");
title("Predictor Importance on Y");

%We deem any predictors/columns that have a value greater than 0.006% to be
%important. We therefore filter out to select only these important
%predictors and ignorning the rest. Hence we set out the trainset and
%testset again using the selected predictors above and the target variable
%'y'.
RelevantPredictors = Predictors >0.00006;
RelPredict_train = trainset(:,[RelevantPredictors true]);
RelPredict_test = testset(:,[RelevantPredictors true]);

%'fitceoc' is used to return a trained ECOC model using the predictors X and the
%class labels Y. This method is combined with hyperparameter optimisation techniques.
%Tree template is used to return default boosted and bagged decision trees for 
%training an ensemble learner template for training an ensemble or (ECOC) multiclass model.
%rng is used in our model which is used to generate random numbers. Seed of
%rng is set to zero if matlab restarts hence the same order is preserved.
%tallrng positions the tall array calculations of the random numbers to
%their default values. tic-toc function is used to record the time.
rng('default')
tallrng('default')
TT = templateTree();
tic; 

%MinLeafSize is used for earlystopping growth of the tree by tuning and hence
%preventing overfitting. 'auto' function is used to use MinLeafSize in our
%tree template. Binning is applied to increase the training speed as we
%have a large dataset. When binning using fitcecoc/tree learner model, we
%allows the data to create specified bins with predictor values and the
%trees grow on these bin indices as oppose to the original dataset. Hence
%this increases the speed of training but at the expense of accuracy. We
%have chosen to bin 52 times as to not compromise on our accuracy much.We
%combine this method with parallel computing (fitcecoc can do this) to also 
%speed up computation.
%onevsrest classification is used to split multiple classes into a single
%binary classification problem per class hence we have one class as + and
%rest as -. This allows us to achieve all possible class combinations.
%Find hyperparameters that minimize five-fold cross-validation loss by 
%using automatic hyperparameter optimization. For reproducibility, 
%the random seed is set and the 'expected-improvement-plus' acquisition
%function is used.
stats = statset('UseParallel',true);
Mdl = fitcecoc(RelPredict_train,'y','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
struct('AcquisitionFunctionName','expected-improvement-plus'),'Learners',TT,...
'NumBins',52,'Options',stats);
toc;

%Calculation of losses on training and test dataset
Treeloss_train = resubLoss(Mdl); 
Treeloss_test = loss(Mdl,RelPredict_test); 

%Calculation for Threshold,AUC,ROC and label predictions 
[EOCX,EOCY,EOCT,EOCAUC,EOCPRED] = fnAucRocValue(Mdl,RelPredict_train); 
[EOCX_TST,EOCY_TST,EOCT_TST,EOCAUC_TST,EOCPRED_TST] = fnAucRocValue(Mdl,RelPredict_test); 

%Calculating model accuracy,misclassification rate, precision, recall &
%F-score
[EOCACCURACY,EOCTMISSCLASS,EOCTPRECISION,EOCTRECALL,EOCTFSCORE] = fnAcc(EOCPRED,RelPredict_train.y); 
[EOCACCURACY_TEST,EOCTMISSCLASS_TEST,EOCTPRECISION_TEST,EOCTRECALL_TEST,EOCTFSCORE_TEST] = fnAcc(EOCPRED_TST,RelPredict_test.y);

%ROC Graph
figure('Name','ROC Comparisons')
plot(EOCX_TST,EOCY_TST) 
hold on
plot(EOCX,EOCY) 
legend("ROC on Test dataset","ROC on Train dataset")
xlabel('False Positive') ;
ylabel('True Positive');
title('ROC Curves for Decision Tree')
hold off

%Graph on Test dataset for accuracy,misclassification rate, precision, recall &
%F-score
Values = [EOCACCURACY_TEST,EOCTPRECISION_TEST,EOCTRECALL_TEST,EOCTFSCORE_TEST,EOCTMISSCLASS_TEST];
PerformanceMetrics = categorical(["Accuracy","Precision","Recall","Fscore","Misclassified rate"]);
figure('Name','Performance Metrics Plot')
bar(PerformanceMetrics,Values)
title("Performance Metrics Plot for Decision Tree");

%Confusion matrix on the final optimised model with Test dataset
figure('Name','Decision Tree Confusion Matrix')
confusionchart(RelPredict_test.y,EOCPRED_TST)
title("Performance Metrics Plot for Decision Tree");

% Formulas to calculate accuracy,misclassification rate, precision, recall &
%F-score
function [Accuracy,Misclassif,Precision,Recall,Fscore] = fnAcc(Predictions,RealLabels)
CorrLabPred = Predictions==RealLabels; %Correct Label Prediction
InCorrLabPred = Predictions~=RealLabels; %Incorrect Label Prediction
Accuracy = sum(CorrLabPred)/numel(Predictions);
Misclassif = sum(InCorrLabPred)/numel(Predictions); %Misclassification Rate

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
function [X,Y,T,AUC,Label] = fnAucRocValue(mdl,dataset)
[Label,Score] = predict(mdl,dataset);
[X,Y,T,AUC] = perfcurve(dataset.y,Score(:,mdl.ClassNames),'true');
end