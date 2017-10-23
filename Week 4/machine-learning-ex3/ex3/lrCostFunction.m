function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

    % Initialize some useful values
    m = length(y); % number of training examples
    num_features = size(X, 2);

    h_theta_x = sigmoid(X * theta);

    J = (sum((-1 * y) .* log(h_theta_x) - (1 - y) .* log(1 - h_theta_x)) / m) + ...
        sum(theta(2:end) .^ 2) * lambda / (2 * m);

    gradient_non_regularized = sum(repmat(h_theta_x - y, 1, num_features) .* X) / m;

    grad = gradient_non_regularized';

    penalty_regularized = theta * (lambda / m);
    grad(2:end) = grad(2:end) + penalty_regularized(2:end);

end
