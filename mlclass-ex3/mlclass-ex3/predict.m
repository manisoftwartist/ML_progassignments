function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add column of ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for sample_no = 1:m
      a1 = X(sample_no,:);			% First layer = inputs
      % Hidden layer
      z2 = Theta1 * a1';       
      a2 = sigmoid(z2);			% Output of hidden layer - inputs to last layer
      a2 = [1;a2];				% a2_0 = 1
      % Third layer
      z3 = Theta2 * a2;
      a3 = sigmoid(z3);			% a3 should be 10x1 column vector, the prediction of the last layer
      A(sample_no,:) = a3';
endfor

% When you exit the above loop, the size of matrix A should be no_of_samplesxno_of_classes(10 here)
% Which means each row has 10 values corresponding to the output for K=1O,total no of classes that we have. The index of max of each row is the class we predict
[val, p] = max(A,[],2);
% Each row of A is a prediction for 1 sample






% =========================================================================


end
