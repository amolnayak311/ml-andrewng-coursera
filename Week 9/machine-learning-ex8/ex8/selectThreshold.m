function [bestEpsilon, bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

    bestEpsilon = 0;
    bestF1 = 0;    
    stepsize = (max(pval) - min(pval)) / 1000;
    for epsilon = min(pval):stepsize:max(pval)
        is_outlier = pval < epsilon;
        tp = sum(is_outlier == 1 & yval == 1);
        fp = sum(is_outlier == 1 & yval == 0);        
        fn = sum(is_outlier == 0 & yval == 1);
        precision = tp / (tp + fp);
        recall = tp / (tp + fn);
        F1 = (2 * precision * recall) / (precision + recall);
    
        if F1 > bestF1
           bestF1 = F1;
           bestEpsilon = epsilon;
        end
    end

end
