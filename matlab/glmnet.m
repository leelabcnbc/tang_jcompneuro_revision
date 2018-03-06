function [B, FitInfo] = glmnet(X, y, standardize, family, alpha)
%GLMNET very simple glmnet. simply do fitting at 100 lambdas.

[N, ~] = size(X);
assert(ismatrix(X));
assert(isequal(size(y), [N, 1]));


if isequal(family, 'poisson')
    [B, FitInfo] = lassoglm_fix(X, y, 'poisson', 'Alpha', alpha, ...
        'Standardize', standardize, ...
        'MaxIter', 1e6, 'RelTol', 1e-6);
elseif isequal(family, 'gaussian')
    [B, FitInfo] = lassoglm_fix(X, y, 'normal', 'Alpha', alpha, ...
        'Standardize', standardize, ...
        'MaxIter', 1e6, 'RelTol', 1e-6);
else
    assert(isequal(family, 'softplus'));
    [B, FitInfo] = lassoglm_fix(X, y, 'normal', 'Alpha', alpha, ...
        'Standardize', standardize, 'Link', softplus_handles(), ...
        'MaxIter', 1e6, 'RelTol', 1e-6);
end
end
