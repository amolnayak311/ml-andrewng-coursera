function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
    m = size(X, 1);
    grads = arrayfun(@(x) ...
        trainLinearReg(X(1:x, :), y(1:x), lambda), ...
        1:m,...
        'UniformOutput', 0);
    % You need to return these values correctly
    
    error_train = arrayfun(@(x) computeError(X(1:x, :), y(1:x), grads{x}), 1:m);
    error_val   = arrayfun(@(x) computeError(Xval, yval, grads{x}), 1:m);
end

%
%
%
function error = computeError(X, y, theta)
    error = sum(((X * theta) - y) .^ 2) / (2 * numel(y));
end
