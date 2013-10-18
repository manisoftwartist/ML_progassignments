function Iprof = gray_intensity_profile(Npix)
%INTENSITY_PROFILE Generate sigmoid intensity profile from 0 to 1 
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = ones(size(z))./(1+exp(-z));

% =============================================================

end
