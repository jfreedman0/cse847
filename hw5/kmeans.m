clear

% options
num_points = 250;
num_clusters = 5;
dimensions = 2; % must be at least 2, more than 2 is hard to visualize
epsilon = 0.000001; % stopping condition threshold (difference in centroids)
iters = 2000; % max iterations
a = 0; % min for random numbers
b = 100; % max for random numberes

% create random points between a and b
data = a + (b-a).*rand(num_points,dimensions);

% pick random centroids to start
centroid_indx = randi(num_points,num_clusters,1);
centroids = data(centroid_indx,:);

% add column to data for cluster assignment
data = horzcat(data, zeros(num_points,1));
[m,n] = size(data);

for i=1:iters
    % for each point, assign it
    for j=1:m 
        point = data(j,1:dimensions);
        
        c_dist = zeros(1,num_clusters);
        for c=1:num_clusters
            c_dist(c) = norm(centroids(c,:)-point);
        end
        [min_val, min_indx] = min(c_dist);
        data(j,dimensions+1) = min_indx;
    end
    
    prev_centroids = centroids;
    % for each centroid, update it
    for c=1:num_clusters
        cluster_indx = data(:,dimensions+1) == c;
        cluster = data(cluster_indx, 1:dimensions);
        centroids(c,:) = mean(cluster);
    end
    
    if  mean(abs(centroids-prev_centroids)) < epsilon
        i; % show number of iterations
        break
    end
end

% display our clusters
figure
scatter(centroids(:,1), centroids(:,2), 1500, "+", "black");
hold on
for c=1:num_clusters
    cluster_indx = data(:,dimensions+1) == c;
    cluster = data(cluster_indx, 1:dimensions);
    scatter(cluster(:,1), cluster(:,2), "filled");
end

data;