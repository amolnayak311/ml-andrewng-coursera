function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

    % Unfold the U and W matrices from params
    X = reshape(params(1:num_movies*num_features), num_movies, num_features);
    Theta = reshape(params(num_movies*num_features+1:end), ...
                    num_users, num_features);

    % You need to return the following values correctly
    predicted_ratings_sq = (X * Theta' - Y).^ 2;
    predicted_ratings_sq(~R) = 0;
    J = sum(predicted_ratings_sq(:)) / 2 + (sum(sum(X .^ 2)) + sum(sum(Theta .^ 2))) * lambda / 2;
    
    
    predicted_ratings = ((X * Theta') - Y);
    predicted_ratings(~R) = 0;
    Theta_grad = predicted_ratings' * X + (lambda * Theta);
    X_grad = predicted_ratings * Theta + (lambda * X);
    %X_grad = zeros(size(X));
    %Theta_grad = zeros(size(Theta));
    grad = [X_grad(:); Theta_grad(:)];
end
