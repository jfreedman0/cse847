load("diabetes.mat");

% Add column of 1s to training data
[m, n] = size(x_train);
x_train = horzcat(ones(m,1), x_train);
 
% Add column of 1s to testing data
[m, n] = size(x_test);
x_test = horzcat(ones(m,1), x_test);
 
lambdas = [1e-5; 1e-4; 1e-3; 1e-2; 1e-1; 1; 10];
[m, n] = size(lambdas);

training_errors = zeros(m,1);
testing_errors = zeros(m,1);

for i = 1:m
    w_ridge = ridge_regression(x_train, y_train, lambdas(i));
    
    predict_train = x_train*w_ridge;
    training_errors(i) = compute_mse(y_train, predict_train);
    
    predict_test = x_test*w_ridge;
    testing_errors(i) = compute_mse(y_test, predict_test);
end

% 5-fold cross validation
% Based on https://www.mathworks.com/help/bioinfo/ref/crossvalind.html

k = 5;
indices = crossvalind('KFold', y_train, k);
cv_error_lambda = zeros(m,1);

for i = 1:m % for each lambda
    cv_error_fold = zeros(k,1);
    for j = 1:k % for each fold
        test = (indices == j);
        train = ~test;
        w_ridge = ridge_regression(x_train(train,:), y_train(train,:), lambdas(i));

        predict_fold_test = x_train(test,:)*w_ridge;
        cv_error_fold(j) = compute_fold_cv_error(x_train(test,:), y_train(test,:), predict_fold_test);
    end
    cv_error_lambda(i) = compute_overall_cv_error(cv_error_fold);
end

% Find minimum cv error and index
[min_error, min_index] = min(cv_error_lambda);

% % loglog for log scale plots
% https://www.mathworks.com/help/matlab/ref/loglog.html

loglog(lambdas, training_errors)
hold on
loglog(lambdas, testing_errors)
hold on
% vertical line code from
% https://www.mathworks.com/matlabcentral/answers/20179-how-to-draw-line-vertical-to-y-axis
yL = get(gca,'YLim');
line([lambdas(min_index) lambdas(min_index)],yL,'Color','black','LineStyle','--');

% Add labels to plot
xlabel('Lambda value')
ylabel('MSE')
legend({'Training Error','Testing Error', '5-Fold best lambda'},'Location','northwest')

% Functions

% Ridge regression solver formula from slides
function w_ridge = ridge_regression(X, y, l)
    [m, n] = size(X);
    w_ridge = inv(transpose(X)*X + l*eye(n))*transpose(X)*y;
end

function mse = compute_mse(ground, predicted)
    [m, n] = size(ground);
    sum = 0;
    for i = 1:m
        sum = sum + (ground(i)-predicted(i))^2;
    end
    
    mse = (1/m)*sum;
end

function fold_cv_error = compute_fold_cv_error(dk, ground, predicted)
    [m, n] = size(ground);
    sum = 0;
    for i = 1:m
        sum = sum + (ground(i)-predicted(i))^2;
    end
    
    [m, n] = size(dk);
    fold_cv_error = inv(m)*sum;
end

function overall_cv_error = compute_overall_cv_error(cv_error_fold)
    [m, n] = size(cv_error_fold);
    overall_cv_error = inv(m)*sum(cv_error_fold);
end

