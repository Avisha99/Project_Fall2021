clc;
clear all;
close all;
dataset=load("naive3.csv");
Insulin=dataset(:,1);
BMI=dataset(:,2);
Outcome=dataset(:,3);
X=[Insulin,BMI];
y=Outcome;
plot(Insulin,BMI,'*')
figure;
%feature scaling
stand_Insulin = (Insulin - mean(Insulin))/std(Insulin);
Insulin = stand_Insulin;
stand_BMI = (BMI - mean(BMI))/std(BMI);
BMI= stand_BMI;
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
Insulin_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
BMI_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(Insulin_range,BMI_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
Insulin_train=dataTrain(:,1);
BMI_train=dataTrain(:,2);
Y = ismember(dataTrain(:,3),0);
scatter(Insulin_train(Y),BMI_train(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(Insulin_train(~Y),BMI_train(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('Insulin');
ylabel('BMI');
title(classifier_name);
legend off, axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off
figure;
labels = unique(Outcome);
classifier_name = 'Naive Bayes (Testing)';
Insulin_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
BMI_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(Insulin_range,BMI_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
Y = ismember(dataTest(:,3),0);
Insulin_test=dataTest(:,1);
BMI_test=dataTest(:,2);
scatter(Insulin_test(Y),BMI_test(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(Insulin_test(~Y),BMI_test(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('Insulin');
ylabel('BMI');
title(classifier_name);
legend off,axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off