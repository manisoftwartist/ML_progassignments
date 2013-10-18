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
%         that your implementation is correct by running checkNNGradients
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
%

% -------------------------- Manikandan.B my code begins ------------------------------------------------
% ------------------------
% Part1 - Cost function 
% ------------------------
X = [ones(m, 1) X];				% Column of ones
A2 = sigmoid( X * Theta1');			% Activation for all training examples (each row for 1 example)
A3 = sigmoid([ones(m, 1) A2] * Theta2');	% Activation of 3rd layer = output layer (each row is output 'y' vector for 1 example)
						% A3 is a matrix (No.egs x No. of classes)
						% Each row of A3 is h_theta(x_i), i.e., prediction for example i			   
for i = 1:m % Loop over training examples
        yvec_i = 1:num_labels;
        % This step is for K classes;generalization of LR cost function (See formula in slides of video lecture 9 NN learning)
	% Instead of for loop over no. of classes, we vectorize
        yvec_i = (yvec_i == y(i));	% y component corres. to class 'k' = 1 ; uses logical arrays        
	h_theta_xi = A3(i,:);	
	innersummationvec = (yvec_i.*log(h_theta_xi)) + ( (1-yvec_i).*(log(1-h_theta_xi)) ); 
	% each element in this vector is one element of the summation term of the cost function
	innersum(i) = sum(innersummationvec);
endfor
J = (-1/m)*sum(innersum);	

% Regularization
% Drop first column corres. to bias terms
thet1 = Theta1(1:end,2:end);
thet2 = Theta2(1:end,2:end);

% ------- Required only if using for loop 
%L = 3;					% No. of layers can be assumed as per assignment;This is network architecture
%[sl(1), slplus1(1)] = size(thet1);	% sl(1) - No of units in layer 1
%[sl(2), slplus1(2)] = size(thet2);
% ------- Required only if using for loop 

thetsqrd = sum(sum(thet1.^2)) + sum(sum(thet2.^2));
regterm = (lambda/(2*m)) * thetsqrd;
J = J + regterm;

% -----------------------------
% ---- Part2 BACKPROPOGATION
% -----------------------------
DELTA_layer2 = zeros(size(Theta2));
DELTA_layer1 = zeros(size(Theta1));

for t = 1:m 
	% Steps as suggested in the assignment
	% STEP 1
	a1 = [X(t,:)]';
	% Hidden layer
	z2 = Theta1 * a1;       
	a2 = sigmoid(z2);			% Output of hidden layer - inputs to last layer
	a2 = [1;a2];				% a2_0 = 1
	% Third layer
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);			% a3 should be 10x1 column vector, t

	% STEP 2
	% Error predictions for last layer
	yvec_i = 1:num_labels;
        yvec_i = (yvec_i == y(t));	% Logical array 1 at class of example t, zero everywhere
	% For each unit in layer 3 (vectorized)
	yvec_i = yvec_i';	
	del_layer3 = a3 - yvec_i;
	%fprintf('dellayer3 size: %d %d\n',size(del_layer3));

	% STEP 3
	% For hidden layer 
	siggrad = sigmoidGradient(z2);
	del_layer2 = (thet2'*del_layer3).*siggrad;
	% del_layer2 = del_layer2(2:end); % del_layer2(0) is removed
	%fprintf('dellayer2 size: %d %d\n',size(del_layer2));

	% STEP 4 - Gradient Accumulation
	% For calculating the Partial Derivative
	% DELTA_layer2 is kind of the partial derivative
	% matrix of the weights(Theta) corresponding to layer 2
	% Note that they are of the same size as the respective Theta
	% Storing in the third dimension, which is the layer number
	% First 2 dimensions are rows and matrices of resp. Theta
	DELTA_layer2 = DELTA_layer2 + (del_layer3 * a2');
	DELTA_layer1 = DELTA_layer1 + (del_layer2 * a1');
	
endfor

% -------------------------------------------------------------
% Gradients D_ij
Theta1_grad = (1/m)*DELTA_layer1;
Theta2_grad = (1/m)*DELTA_layer2;
% =========================================================================
% thet1 and thet2 (columns are dropped)
reg_term_Theta1_grad = (lambda/m)*thet1;
reg_term_Theta2_grad = (lambda/m)*thet2;

Theta1_grad = Theta1_grad + [zeros(size(Theta1_grad,1),1) reg_term_Theta1_grad]; 
Theta2_grad = Theta2_grad + [zeros(size(Theta2_grad,1),1) reg_term_Theta2_grad];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
