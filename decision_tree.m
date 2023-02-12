clc;
clear all;
close all;
data=load('diabetes1.csv');
Pregnancies=data(:,1);
Glucose=data(:,2);
BloodPressure=data(:,3);
SkinThickness=data(:,4);
Insulin=data(:,5);
BMI=data(:,6);
DiabetesPedigreeFunction=data(:,7);
Age=data(:,8);
Outcome=data(:,9);
% feature scaling
Pregnancies=(Pregnancies-min(Pregnancies))/(max(Pregnancies)-min(Pregnancies));
Glucose=(Glucose-min(Glucose))/(max(Glucose)-min(Glucose));
BloodPressure=(BloodPressure-min(BloodPressure))/(max(BloodPressure)-min(BloodPressure));
SkinThickness=(SkinThickness-min(SkinThickness))/(max(SkinThickness)-min(SkinThickness));
Insulin=(Insulin-min(Insulin))/(max(Insulin)-min(Insulin));
BMI=(BMI-min(BMI))/(max(BMI)-min(BMI));
DiabetesPedigreeFunction=(DiabetesPedigreeFunction-min(DiabetesPedigreeFunction))/(max(DiabetesPedigreeFunction)-min(DiabetesPedigreeFunction));
Age-(Age-min(Age))/(max(Age)-min(Age));
%divide data into training and testing
cv = cvpartition (768,'HoldOut',0.2);
X1_train= Pregnancies(training(cv));
X2_train= Glucose(training(cv));
X3_train= BloodPressure(training(cv));
X4_train= SkinThickness(training(cv));
X5_train= Insulin(training(cv));
X6_train= BMI(training(cv));
X7_train= DiabetesPedigreeFunction(training(cv));
X8_train= Age(training(cv));
Y_train= Outcome(training(cv));
X1_test= Pregnancies(test(cv));
X2_test = Glucose(test(cv));
X3_test = BloodPressure(test(cv));
X4_test = SkinThickness(test(cv));
X5_test = Insulin(test(cv));
X6_test = BMI(test(cv));
X7_test = DiabetesPedigreeFunction(test(cv));
X8_test = Age(test(cv));
Y_test = Outcome(test(cv));
data_train=[X1_train,X2_train,X3_train,X4_train,X5_train,X6_train,X7_train,X8_train,Y_train];
data_test=[X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test,X8_test,Y_test];
%train model
classification_model=fitctree([X1_train,X2_train,X3_train,X4_train,X5_train,X6_train,X7_train,X8_train],Y_train);
testset=[X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test,X8_test];
testLabels=Y_test;
testresult=predict(classification_model,testset);
%draw confusion matrix
C=confusionmat(testLabels,testresult);
disp(C);
%calculate accuracy
accuracy=sum((testresult==testLabels)/length(testLabels)*100);
fprintf(" Accuracy = %f\n" , accuracy);
view(classification_model,'mode','graph')

