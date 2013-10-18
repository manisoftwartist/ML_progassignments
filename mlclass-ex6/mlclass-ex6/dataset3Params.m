function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% Values to try
C_try = [0.01, 0.03, 0.1,0.3, 1,3,10,30];
sigma_try = C_try;

u = size(C_try,2);
v = size(sigma_try,2);

ERR_vector = ones(u,v);

for i = 1:u
	C = C_try(i);
   for j = 1:v   	
   	sigma = sigma_try(j);
	model = svmTrain(X, y,C , @(x1, x2) gaussianKernel(x1, x2, sigma));    
	predictions = svmPredict(model, Xval);
	ERR_vector(i,j) = mean(double(predictions ~= yval));
	if (i==1 && j==1)
	   minvalue = ERR_vector(1,1);
	   % indices for minimum value
	   idx1 = 1; 
	   idx2 = 1; 
	endif
	if ERR_vector(i,j) < minvalue
	   minvalue = ERR_vector(i,j);
	   idx1 = i;
	   idx2 = j;
   	endif
   endfor
endfor

C = C_try(idx1);
sigma = sigma_try(idx2);
% =========================================================================

end
