function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


% -----------------------------------------------
% >>>>>>>>>>>>>>>>>>>> PART 1 <<<<<<<<<<<<<<<<<<<
% -----------------------------------------------

% ------- FORWARD PROPAGATION ALGORITHM

% First, we have to calculate HThetaX = a3
X = [ones(m,1) X]; %adding bias to all training sets. size: 5000x401
z2 = X * Theta1'; % X(5000 x 401), Theta1^T (401 x 25)
a2 = sigmoid(z2); % a2(5000 x 25)
a2 = [ones(m,1) a2]; % a2(5000 x 26)

z3 = a2 * Theta2'; % a2(5000 x 26), Theta2^T (26 x 10)
a3 = sigmoid(z3); % a3(5000 x 10) hThetaX
hThetaX = a3;

% --- Make the yVector. y is a vector just with the numbers. e.g [5,6,7,2,3,9,0...]

yVector = zeros(m,num_labels); % yVector(m training examples x k classes) e.g 5000x10
for i=1:m
	yVector(i,y(i,1)) = 1;
end

% --- Calculate J(theta) with the formula
result = yVector .* log(hThetaX)+ (1-yVector) .* log(1-hThetaX); % formula to get J(theta)
s = 0;
for i=1:m
	s = s + sum(result(i,:));
end
J = (-1/m) * s;

% --- Regularize J(theta)
% we do not regularize the biat units.
Theta1WithoutBias = Theta1(:,2:size(Theta1,2));
Theta2WithoutBias = Theta2(:,2:size(Theta2,2));

regularization = lambda/(2*m) * (sum(sum(Theta1WithoutBias .^ 2)) + sum(sum(Theta2WithoutBias .^ 2)) ); % we have to sum the L layers 
J = J + regularization;

% -----------------------------------------------
% >>>>>>>>>>>>>>>>>>>> PART 2 <<<<<<<<<<<<<<<<<<<
% -----------------------------------------------

% Implement the backpropagation algorithm to compute
% the partial derivatives and the gradients
% Theta1_grad and Theta2_grad

for i = 1:m,
    % Perform forward propagation and backpropagation using example (x(i),y(i))
	% (Get activations a(l) and delta terms d(l) for l = 2,...,L
	
	% Step 1 Getting the input layer
	a1 = (X(i,:))'; % Transpose 401x1 input layer of the i training example
	% Step 2 perform forward propagation to compute all a(l)
	z2 = Theta1*a1; % Theta1 25x401, a(1) 401x1, result 25x1
	a2 = [1; sigmoid(z2)]; % 26x1, we add bias;
	
	z3 = Theta2*a2; % Theta2 10 x 26, a(2) 26x1
	a3 = sigmoid(z3); % output layer
	
	y_i = yVector(i,:)';
	delta3 = a3 - y_i; % 10 x 1
	z2 = [1;z2]; % we add bias
	delta2 = (Theta2' * delta3) .* sigmoidGradient(z2); % (Theta2 Transpose 26x10 delta3 10x1) = 26x1,  sigmoidGradient(z2) 26x1
	delta2 = delta2(2:end); % we get rid of the value calculated with the bias
	
	Theta2_grad = Theta2_grad + delta3 * a2'; % Theta2_grad 10x26, (delta3 10x1, a2 Transpose 1x26) = 10x26
	Theta1_grad = Theta1_grad + delta2 * a1'; % Theta1_grad 25x401, (delta2 25x1, a1 Transpose 1x401) = 25x401

end



% -----------------------------------------------
% >>>>>>>>>>>>>>>>>>>> PART 3 <<<<<<<<<<<<<<<<<<<
% -----------------------------------------------
% regularization with the cost function and gradients.
% Calculating Dij(l)


Theta1_grad = (1/m) * Theta1_grad; % formula for j=0, 25x401
Theta2_grad = (1/m) * Theta2_grad; % formula for j=0, 10x26
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m * Theta2(:,2:end)); % formula for j!=0
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m * Theta1(:,2:end)); % formula for j!=0

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)]; % partial derivate of J(theta)

end
