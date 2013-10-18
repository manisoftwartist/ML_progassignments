function [J, grad] = lrCostFunction(theta, X, y, lambda)

%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Use this for testing
%load('ex3data1.mat');
%[m,n]= size(X);
%X = [ones(m, 1) X];
%theta = zeros(n+1,1); 
%lambda = 0.2;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Cost function 
h_theta = zeros(m,1);    				   % Should be of the same size as y vector
theta_into_xi = X * theta; 				  
h_theta = sigmoid(theta_into_xi);                          % Hypothesis vector i.e., h(theta) for all training examples
							   % Note that this is different from the hypothesis for linear regression 
summationvec = (-y.*log(h_theta)) - ( (1-y).*(log(1-h_theta)) ); % each element in this vector is one element of the summation term of the cost function
% each element of y above is for one training example
% each element of h_theta above is also for 1 training example 
J = (1/m)*sum(summationvec);
%fprintf('J before regul: %f\n',J);

% Adding regularization
theta_exclude_firstval = theta(2:end,1); 	% remember xo is always 1. So theta0 should n t be squared
theta_sqrd = theta_exclude_firstval.^2;		% Note the regularization term starts from 1
J_regterm = (lambda/(2*m))*sum(theta_sqrd);
J = J + J_regterm;
%fprintf('J after regul: %f\n',J);

% Now the gradient
grad = (1/m)*X'*(h_theta-y);
%fprintf('grad before regul: %f\n',grad) 

reg_term = (lambda/m)*theta_exclude_firstval;	
reg_term = [0;reg_term];	                   
grad = grad + reg_term;
% fprintf('grad after regul: %f\n',size(grad,1));

% =============================================================
grad = grad(:);

end
