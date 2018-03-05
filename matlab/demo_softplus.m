rng default % For reproducibility
X = randn(1000,20);
mu = X(:,[5 10 15])*[.4;.2;.3]-5;
y = funcs.softplus(mu);
% scatter(mu, y);
% pause;
% y = mu;
% 5 fold.
foldid = zeros(1000,1);
foldid(1:200) = 1;
foldid(201:400) = 2;
foldid(401:600) = 3;
foldid(601:800) = 4;
foldid(801:1000) = 5;
[predicted_y, B, FitInfo, devsum] = glmnet_cv_best_result(X, y, ...
    false, 'softplus', foldid, 1, true);

close all;
figure;
scatter(y, predicted_y);
hold on;
plot([min(y), max(y)], [min(y), max(y)]);
axis('equal');
hold off;

disp([devsum - FitInfo.Deviance(FitInfo.IndexMinDeviance)]);
