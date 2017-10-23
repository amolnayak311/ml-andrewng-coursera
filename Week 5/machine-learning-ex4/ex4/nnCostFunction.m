function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));


    % Setup some useful variables
    m = size(X, 1);
    X = [ones(m, 1) X];

    predicted_vals = sigmoid([ones(m, 1) sigmoid(X * Theta1')] * Theta2');
    new_y = cell2mat(arrayfun(@(x) to_vec(x, num_labels), y, 'UniformOutput', false));

    J = -sum(sum((new_y) .* log(predicted_vals) + (1 - new_y) .* log(1 - predicted_vals))) / m;

    regularized_cost = sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2 (:, 2:end) .^ 2));

    J = J + (0.5 * lambda * regularized_cost / m);

    %Theta1_grad = zeros(size(Theta1));
    %Theta2_grad = zeros(size(Theta2));

    a1 = X;
    z2 = a1 * Theta1';
    a2 = [ones(m, 1), sigmoid(z2)];

    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    delta3 = a3 - new_y;


    delta2 = (delta3 * Theta2) .* sigmoidGradient([ones(m, 1), z2]);
    delta2 = delta2(:, 2:end);

    Delta1 = delta2' * a1;
    Theta1_grad = Delta1 / m;

    regularized_multiplier = lambda / m;
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (Theta1(:, 2:end) * regularized_multiplier);

    Delta2 = delta3' * a2;
    Theta2_grad = Delta2 / m;
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (Theta2(:, 2:end) * regularized_multiplier);

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

%
%
%
function vec = to_vec(n, num_features)
    vec = zeros(1, num_features);
    vec(n) = 1;
end
