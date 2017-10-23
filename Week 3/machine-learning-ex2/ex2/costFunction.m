function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
    m = length(y); % number of training examples

    h_theta = sigmoid(X * theta);
    J = sum(-y .* log(h_theta) - (1 - y) .* log(1 - h_theta)) / m;
    grad = sum(repmat((h_theta - y), 1, 3) .* X) / m;
end
