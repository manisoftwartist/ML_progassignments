function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = zeros(m,1);    		% Should be of the same size as y vector
theta_into_xi = X * theta;		% Hypothesis vector i.e., h(theta) for all training examples
h_theta = sigmoid(theta_into_xi);       % Note that this is different from the hypothesis for linear regression 
summationvec = -y.*log(h_theta) - (1-y).*(log(1-h_theta)); % each element in this vector is one element of the summation term of the cost function
J = (1/m)*sum(summationvec);

theta_exclude_firstval = theta(2:end,1); % remember xo is always 1. So theta0 should n t be squared
theta_sqrd = theta_exclude_firstval.^2;
regularisation_term = (lambda/(2*m)) * sum(theta_sqrd);
J = J + regularisation_term;


% Gradient part
for j = 1 : size(grad,1)
    sumvector = (h_theta - y).*X(:,j);
    if (j==1)
    	grad(j) = (1/m)*sum(sumvector);
    else
        grad(j) = (1/m)*sum(sumvector) + (lambda/m)*theta(j);
    endif
endfor




% =============================================================

end
