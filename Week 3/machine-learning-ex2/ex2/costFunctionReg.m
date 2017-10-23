function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

    m = length(y); % number of training examples
    num_features = size(X, 2);
    h_thetax = sigmoid(X * theta);
    minus_y = -1 * y;
    J = (sum((minus_y) .* log(h_thetax) - (minus_y + 1) .* log(1 - h_thetax)) / m) + ...
        lambda * sum(theta(2:end) .^ 2) / (2 * m);

    grad_normal = sum(repmat((h_thetax - y), 1, num_features) .* X) / m;
    grad_theta_penalty = zeros(size(grad_normal));
    grad_theta_penalty(2:end) = theta(2:end) * lambda / m;
    grad = (grad_normal + grad_theta_penalty)';
end
