clc;
clear all;
close all;
data=load('diabetes1forneuralnetwork.csv');
X=data(:,[1:8]);
y=data(:,9);
m =  size(X, 1);
cv=cvpartition(768,'Holdout',0.20);
data_train=data(training(cv),:);
data_test=data(test(cv),:);
X1_test=data_test(:,1);
X2_test=data_test(:,2);
X3_test=data_test(:,3);
X4_test=data_test(:,4);
X5_test=data_test(:,5);
X6_test=data_test(:,6);
X7_test=data_test(:,7);
X8_test=data_test(:,8);
Y_test=data_test(:,9);
X_test=[X1_test X2_test X3_test X4_test X5_test X6_test X7_test X8_test];
input_layer_size =  8;  
hidden_layer_size = 25; 
num_labels = 20;

lr=0.1
lambda=1;
maxiterations=3000;
% size(Theta1_grad)
initial_Theta1  =  randInitializeWeights(input_layer_size,  hidden_layer_size);
initial_Theta2  =  randInitializeWeights(hidden_layer_size,  num_labels);
J_cost=zeros(1,maxiterations); 
for  i=1:maxiterations 
    [J, Theta1_gradient,  Theta2_gradient]  =  nnCostFunction(initial_Theta1,initial_Theta2, num_labels, X,  y, lambda); 
    initial_Theta1  = initial_Theta1  -lr.*Theta1_gradient; 
    initial_Theta2  = initial_Theta2  -lr.*Theta2_gradient; 
    J_cost(i)=J;
end

pred =  predict(initial_Theta1,  initial_Theta2, X);
fprintf('\nTraining Set  Accuracy:  %f\n', mean(double(pred ==  y)) * 100);
plot(1:maxiterations,J_cost,'-');
test_pred=predict(initial_Theta1,initial_Theta2,X_test);
fprintf('\nTesting Set Accuracy:%f\n',mean(double(test_pred==Y_test))*100);
plot(1:maxiterations,J_cost,'-');
