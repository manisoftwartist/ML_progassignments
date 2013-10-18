function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
% You should set J to the cost and grad to the gradient.
%

% vectorized version of linear regression
h_theta = zeros(m,1);    				% should be of the same size as y vector
h_theta = X * theta; 					% Hypothesis vector i.e., h(theta) for all training examples
summation = (h_theta - y)' * (h_theta - y);   		% In vector notation, (h(theta) - y)^2 = (h(theta) - y)' * (h(theta) - y)
J = summation/(2*m);

% Adding regularization
theta_exclude_firstval = theta(2:end,1); 		% remember xo is always 1. So theta0 should n t be squared
theta_sqrd = theta_exclude_firstval.^2;			% Note the regularization term starts from 1
J_regterm = (lambda/(2*m))*sum(theta_sqrd);
J = J + J_regterm;

% Now the gradient
grad = (1/m)*X'*(h_theta-y);
%fprintf('grad before regul: %f\n',grad) 

reg_term = (lambda/m)*theta_exclude_firstval;	
reg_term = [0;reg_term];	                   
grad = grad + reg_term;


% =========================================================================

grad = grad(:);

end
