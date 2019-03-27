load('alzheimers/ad_data.mat')
load('alzheimers/feature_name.mat')

% Add column of 1s to training and testing data
[m, n] = size(X_train);
X_train = horzcat(X_train, ones(m,1));

[m, n] = size(X_test);
X_test = horzcat(X_test, ones(m,1));

params = [1e-8; 0.01; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
[m, n] = size(params);

AUC_vals = zeros(m, 1);
nonzero_weights = zeros(m, 1);

for i = 1:m
    [w, c] = logistic_l1_train(X_train, y_train, params(i));
    predictions = X_test*w;
    nonzero = w ~= 0;
    nonzero_weights(i) = sum(nonzero);
    [X, Y, T, AUC] = perfcurve(y_test, predictions, 1);
    AUC_vals(i) = AUC;
end

transpose(nonzero_weights)
transpose(AUC_vals)

plot(params, AUC_vals)
% Add labels to plot
xlabel('Regularization Parameter')
ylabel('AUC')

figure
plot(params, nonzero_weights)
% Add labels to plot
xlabel('Regularization Parameter')
ylabel('Features Selected')

function [w, c] = logistic_l1_train(data, labels, par)
    % OUTPUT w is equivalent to the first d dimension of weights in logistic train
    % c is the bias term, equivalent to the last dimension in weights in logistic train.
    % Specify the options (use without modification).
    opts.rFlag = 1; % range of par within [0, 1].
    opts.tol = 1e-6; % optimization precision
    opts.tFlag = 4; % termination options.
    opts.maxIter = 5000; % maximum iterations.
    [w, c] = LogisticR(data, labels, par, opts);
end