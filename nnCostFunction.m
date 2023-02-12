function [J Theta1_grad Theta2_grad] = nnCostFunction(Theta1,Theta2,num_labels,X, y, lambda)
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1)); 
Theta2_grad = zeros(size(Theta2)); 
X = [ones(m, 1) X];
hidden1_result = sigmoid(Theta1*(X'));
hidden1_result = [ones(1, m); hidden1_result];
pre = (sigmoid(Theta2*hidden1_result))';
y_onehot = full(ind2vec(y',num_labels));
y_onehot =y_onehot';
J = sum(sum((-y_onehot.*log(pre))-((1-y_onehot).*log(1-pre))))/m +(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);
Theta2_grad(:,1) = (sum(pre-y_onehot)/m)';
Theta2_grad(:,2:end) = ((pre-y_onehot)'*hidden1_result(2:end,:)')./m+(Theta2(:,2:end)*lambda/m);
Theta1_grad(:,1) = sum((((pre-y_onehot)*Theta2(:,2:end))'.*sigmoidGradient(Theta1*(X')))')'./m;
Theta1_grad(:,2:end) = ((((pre-y_onehot)*Theta2(:,2:end))'.*sigmoidGradient(Theta1*(X')))*X(:,2:end))./m+(Theta1(:,2:end)*lambda/m); 
end