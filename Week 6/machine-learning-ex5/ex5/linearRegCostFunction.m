function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad
% Initialize some useful values

    m = length(y); % number of training examples

    unregularized_cost = sum(((X * theta) - y) .^ 2) / (2 * m);
    regularized_cost = lambda * sum((theta(2:end) .^ 2)) / (2 * m);

    J = unregularized_cost + regularized_cost;

    diff_prediction  = repmat(((X * theta ) - y), 1, size(theta, 1));
    
    k = diff_prediction .* X;
    if size(k, 1) ~= 1
        grad = sum(k) / m;
    else
        grad = k / m;
    end
    grad_regularized = (theta(2:end) * lambda / m)';
    grad(2:end) = grad(2:end) + grad_regularized;
    grad = grad(:);

end
