clc;
clear all;
close all;
dataset=load("naive1.csv");
Pregnancies=dataset(:,1);
Glucose=dataset(:,2);
Outcome=dataset(:,3);
X=[Pregnancies,Glucose];
y=Outcome;
plot(Pregnancies,Glucose,'*')
figure;
%feature scaling
stand_Pregnancies = (Pregnancies - mean(Pregnancies))/std(Pregnancies);
Pregnancies = stand_Pregnancies;
stand_Glucose = (Glucose - mean(Glucose))/std(Glucose);
Glucose= stand_Glucose;
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
Pregnancies_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
Glucose_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(Pregnancies_range,Glucose_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
Pregnancies_train=dataTrain(:,1);
Glucose_train=dataTrain(:,2);
Y = ismember(dataTrain(:,3),0);
scatter(Pregnancies_train(Y),Glucose_train(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(Pregnancies_train(~Y),Glucose_train(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('Pregnancies');
ylabel('Glucose');
title(classifier_name);
legend off, axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off
figure;
labels = unique(Outcome);
classifier_name = 'Naive Bayes (Testing)';
Pregnancies_range = min(dataTrain(:,1))-1:0.01:max(dataTrain(:,1))+1;
Glucose_range = min(dataTrain(:,2))-1:0.01:max(dataTrain(:,2))+1;
[xx1, xx2] = meshgrid(Pregnancies_range,Glucose_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(classification_model,XGrid);
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
Y = ismember(dataTest(:,3),0);
Pregnancies_test=dataTest(:,1);
Glucose_test=dataTest(:,2);
scatter(Pregnancies_test(Y),Glucose_test(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(Pregnancies_test(~Y),Glucose_test(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('Pregnancies');
ylabel('Glucose');
title(classifier_name);
legend off,axis tight
legend('0','1','Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
hold off