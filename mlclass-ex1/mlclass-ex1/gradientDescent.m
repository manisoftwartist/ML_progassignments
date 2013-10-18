function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

jdim = size(X,2);

% Normalize data
k = mean(X);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

       sum_term = zeros(2,1); 			% Value of summation terms for SIMULTANEOUS update of theta1 and theta2    	
       for i = 1:m
   	    h_theta(i) = dot(theta,X(i,:));	% Should remain constant while theta values are updated
       endfor

	for j = 1: jdim
        	for i = 1:m
  			sum_term(j) = sum_term(j) + ( (h_theta(i) - y(i) ) * X(i,j) );
        	endfor
  		theta(j) = theta(j) - ( (alpha /m) * sum_term(j) ) ;              % Simultaneously update both thetas
  	endfor
  	     
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('Cost function J: ');
    fprintf('%f\n', J_history(iter));

endfor

end
