clc;
clear all;
close all;
dataset=load("naive2.csv");
BloodPressure=dataset(:,1);
SkinThickness=dataset(:,2);
Outcome=dataset(:,3);
X=[BloodPressure,SkinThickness];
y=Outcome;
plot(BloodPressure,SkinThickness,'*')
figure;
%feature scaling
stand_BloodPressure = (BloodPressure - mean(BloodPressure))/std(BloodPressure);
BloodPressure = stand_BloodPressure;
stand_SkinThickness = (SkinThickness - mean(SkinThickness))/std(SkinThickness);
SkinThickness= stand_SkinThickness;
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
BloodPressure_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
SkinThickness_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(BloodPressure_range,SkinThickness_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
BloodPressure_train=dataTrain(:,1);
SkinThickness_train=dataTrain(:,2);
Y = ismember(dataTrain(:,3),0);
scatter(BloodPressure_train(Y),SkinThickness_train(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(BloodPressure_train(~Y),SkinThickness_train(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('BloodPressure');
ylabel('SkinThickness');
title(classifier_name);
legend off, axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off
figure;
labels = unique(Outcome);
classifier_name = 'Naive Bayes (Testing)';
BloodPressure_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
SkinThickness_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(BloodPressure_range,SkinThickness_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
Y = ismember(dataTest(:,3),0);
BloodPressure_test=dataTest(:,1);
SkinThickness_test=dataTest(:,2);
scatter(BloodPressure_test(Y),SkinThickness_test(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(BloodPressure_test(~Y),SkinThickness_test(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('BloodPressure');
ylabel('SkinThickness');
title(classifier_name);
legend off,axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off