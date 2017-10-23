function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

    % Selected values of lambda (you should not change this)
    lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

    thetas = arrayfun(@(x) trainLinearReg(X, y, x), lambda_vec, 'UniformOutput', 0);

    % You need to return these variables correctly.
    error_train = arrayfun(@(m) computeError(X, y, thetas{m}), 1:numel(lambda_vec));
    error_val = arrayfun(@(m) computeError(Xval, yval, thetas{m}), 1:numel(lambda_vec));

end

%
%
%
function error = computeError(X, y, theta)
    error = sum(((X * theta) - y) .^ 2) / (2 * numel(y));
end