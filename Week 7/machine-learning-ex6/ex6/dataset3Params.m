function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

    possible_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    n = numel(possible_values);
    cs = repmat(possible_values, 1, n)';

    d = repmat(possible_values, n, 1);
    deltas = d(:);

    models = arrayfun(@(x) svmTrain(X, y, ...
                        cs(x), @(a, b) gaussianKernel(a, b, deltas(x))),...
                   1:(n * n));

    cv_errors = arrayfun(@(model) mean(double(yval ~= svmPredict(model, Xval))), models);
    min_errors = find(cv_errors == min(cv_errors));
    min_error_index = min_errors(1);
    C = cs(min_error_index);
    sigma = deltas(min_error_index);

end
