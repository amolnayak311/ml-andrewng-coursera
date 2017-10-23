function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

    % Set K
    K = size(centroids, 1);

    res = arrayfun(@(x) distance_to_centroid(centroids(x,:), X), 1:K, 'UniformOutput', 0);
    distances = horzcat(res{:});
    [~, idx] = min(distances, [], 2);
end

%
%
%
function distance = distance_to_centroid(centroid, points)
    diff = points - repmat(centroid, size(points, 1), 1);    
    distance = sum(diff .^ 2, 2);
end
