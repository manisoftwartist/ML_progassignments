function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

summation = 0;
h_theta = 0;

for i = 1:m
        h_theta = dot(theta',X(i,:));	% X(i,:) is just the training data for 1 sample (including the x0=1 that we added) 
        %h_theta = X(i,:) * theta;
	summation = summation + (h_theta - y(i)) * (h_theta - y(i)); 
endfor

J = summation/(2*m);

% =========================================================================

end
