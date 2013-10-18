function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


	% First what is your predictor. We have epsilon(in each iteration we want to try)
	% Prediction rule, for every example x(i) , if p(x(i)) < epsilon predict x(i) as anamoly.
	% pval has p(x(i)) for all the "m" examples
	our_algo_predictions = (pval<epsilon);
	
	% True positives. Predictions and groundtruth "yval" both say 1
	tp_vec = (our_algo_predictions == 1) & (yval==1); % LOGICAL "AND"; will leave 1s in the indexes where both preicted and yval are equal to 1
	tp = sum(tp_vec);
	% Similar to above, in 1 step
	fp = sum((our_algo_predictions == 1) & (yval==0));
	fn = sum((our_algo_predictions == 0) & (yval==1));

	prec = tp/(tp+fp);
	recall = tp/(tp+fn);
	F1 = ((2*prec*recall)/(prec+recall));

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
