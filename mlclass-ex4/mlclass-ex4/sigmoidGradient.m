function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

sigmoid_of_z = sigmoid(z); 
one_generic = ones(size(sigmoid_of_z));	% becomes a vector or matrix as per case 
thegradientofsigmoid = sigmoid_of_z .* (one_generic - sigmoid_of_z); 
g = thegradientofsigmoid;


% =============================================================




end
