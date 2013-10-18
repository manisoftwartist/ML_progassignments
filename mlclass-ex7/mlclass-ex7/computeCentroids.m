function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for k = 1:K
	logarr = (idx == k); 					% Use logical arrays to get 1 in places where the entries are equal to centroid index 'k'
	indices = find(logarr); 				% Indices contains the training examples which are assigned to centroid 'k'
	indices = indices';						% Row vector
	X_assigned_k = X(indices,:);					% Logical array indexing
	summation_X_assigned_k = sum(X_assigned_k,1);	% Add along dimension 1, the columns
	N_with_centroid_k = size(indices,2); 			% N = no_of_egs_with_k_as_centroid
	centroids(k,:) = summation_X_assigned_k./N_with_centroid_k;
endfor






% =============================================================


end

