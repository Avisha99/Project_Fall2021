clc;
clear all;
close all;
dataset=load("naive4.csv");
DiabetesPedigreeFunction=dataset(:,1);
Age=dataset(:,2);
Outcome=dataset(:,3);
X=[DiabetesPedigreeFunction,Age];
y=Outcome;
plot(DiabetesPedigreeFunction,Age,'*')
figure;
%feature scaling
stand_DiabetesPedigreeFunction = (DiabetesPedigreeFunction - mean(DiabetesPedigreeFunction))/std(DiabetesPedigreeFunction);
DiabetesPedigreeFunction = stand_DiabetesPedigreeFunction;
stand_Age = (Age - mean(Age))/std(Age);
Age= stand_Age;
classification_model = fitcnb(X,y); 
cv = cvpartition(classification_model.NumObservations, 'HoldOut', 0.2);
dataTrain=dataset(training(cv),:);
dataTest=dataset(test(cv),:);
predictionsTest = predict(classification_model,dataTest(:,[1,2]));
predictionsTrain = predict(classification_model,dataTrain(:,[1,2]));
accuracy_for_trainingset=sum(dataTrain(:,3)==predictionsTrain)/length(predictionsTrain)
accuracy_for_testset=sum(dataTest(:,3)==predictionsTest)/length(predictionsTest)
labels = unique(Outcome);
classifier_name = 'Naive Bayes (Training)';
DiabetesPedigreeFunction_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
Age_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(DiabetesPedigreeFunction_range,Age_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
DiabetesPedigreeFunction_train=dataTrain(:,1);
Age_train=dataTrain(:,2);
Y = ismember(dataTrain(:,3),0);
scatter(DiabetesPedigreeFunction_train(Y),Age_train(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(DiabetesPedigreeFunction_train(~Y),Age_train(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('DiabetesPedigreeFunction');
ylabel('Age');
title(classifier_name);
legend off, axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off
figure;
labels = unique(Outcome);
classifier_name = 'Naive Bayes (Testing)';
DiabetesPedigreeFunction_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
Age_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(DiabetesPedigreeFunction_range,Age_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
Y = ismember(dataTest(:,3),0);
DiabetesPedigreeFunction_test=dataTest(:,1);
Age_test=dataTest(:,2);
scatter(DiabetesPedigreeFunction_test(Y),Age_test(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(DiabetesPedigreeFunction_test(~Y),Age_test(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('DiabetesPedigreeFunction');
ylabel('Age');
title(classifier_name);
legend off,axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off