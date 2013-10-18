function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1) % No of training examples

for i = 1:m
	x_i = X(i,:);
	temp = repmat(x_i,K,1);				% To subtract directly with centroids
	diffr = temp - centroids;
	norm_diffr = norm(diffr,2,"rows");
	sqrd_norm = norm_diffr.^2;
	[dummy,min_index] = min(sqrd_norm);
	idx(i) = min_index;
endfor

% =============================================================

end

