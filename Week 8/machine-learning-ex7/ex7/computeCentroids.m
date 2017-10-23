function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

    masks = arrayfun(@(k) idx == k, 1:K, 'UniformOutput', 0);
    
    segregated_data = cellfun(@(mask) X(mask,:), masks, 'UniformOutput', 0);

    new_centroids = cellfun(@(x) sum(x) / size(x, 1), segregated_data, 'UniformOutput', 0);
    % You need to return the following variables correctly.
    centroids = vertcat(new_centroids{:});
end

