load('spam_email/data.txt')
load('spam_email/labels.txt')

% Add column of 1s to dataset
[m, n] = size(data);
data = horzcat(data, ones(m,1));

% Split into training and testing
x_train = data(1:2000,:);
x_test = data(2001:4601,:);

y_train = labels(1:2000,:);
y_test = labels(2001:4601,:);

training_sizes = [200; 500; 800; 1000; 1500; 2000];
% training_sizes = [500];
[m, n] = size(training_sizes);

accuracies = zeros(m, 1);

for i = 1:m
    weights = log_train(x_train(1:training_sizes(i),:), y_train(1:training_sizes(i),:));
    
    predictions = x_test*weights;
    accuracies(i) = compute_accuracy(y_test, round(sigmf(predictions, [1 0])));
end

transpose(accuracies)

plot(training_sizes, accuracies)
% Add labels to plot
xlabel('Training Size')
ylabel('Accuracy')

function accuracy = compute_accuracy(ground, predicted)
    predicted_correctly = ground == predicted;
    accuracy = sum(predicted_correctly) / length(predicted_correctly);
end

function weights = log_train(data, labels, varargin)
    if nargin == 2
        epsilon = 1e-5;
        maxiter = 1000;
    elseif nargin == 3
        epsilon = varargin{1};
        maxiter = 1000;
    elseif nargin == 4
        epsilon = varargin{1};
        maxiter = varargin{2};
    end
    
    [m, n] = size(data);
    weights = zeros(n, 1);
    
    for i = 1:maxiter
        % logistic sigmoid function docs: https://www.mathworks.com/help/fuzzy/sigmf.html
        prev_iter_predictions = sigmf(data*weights, [1 0]);
        r = diag(prev_iter_predictions.*(1-prev_iter_predictions));
        [m, n] = size(r);
        r = r+0.1*eye(m);
        z = data*weights-inv(r)*(prev_iter_predictions-labels);
        % update weights based on 4.99 from book
        weights = inv(transpose(data)*r*data)*transpose(data)*r*z;
        new_iter_predictions = sigmf(data*weights, [1 0]);
        
        if  mean(abs(new_iter_predictions-prev_iter_predictions)) < epsilon
            break
        end
    end
end