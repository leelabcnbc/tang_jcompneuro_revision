function [lambda] = glmnet_lambda_sequence(X, y, standardize, family, alpha, numlambda)
%GLMNET_CV_BEST_RESULT a wrapper for 1 call of lassoglm with CV
% and 5 calls without to get the raw test response.

% if nargin < 6 || isempty(numlambda)
%     numlambda = 100;
% end

[N, ~] = size(X);
assert(ismatrix(X));
assert(isequal(size(y), [N, 1]));


if isequal(family, 'poisson')
    [lambda] = glmnet_lambda_sequence_inner(X, y, 'poisson', 'Alpha', alpha, ...
        'Standardize', standardize, ...
        'MaxIter', 1e6, 'RelTol', 1e-6, 'NumLambda', numlambda);
elseif isequal(family, 'gaussian')
    [lambda] = glmnet_lambda_sequence_inner(X, y, 'normal', 'Alpha', alpha, ...
        'Standardize', standardize, ...
        'MaxIter', 1e6, 'RelTol', 1e-6, 'NumLambda', numlambda);
else
    assert(isequal(family, 'softplus'));
    [lambda] = glmnet_lambda_sequence_inner(X, y, 'normal', 'Alpha', alpha, ...
        'Standardize', standardize, 'Link', softplus_handles(), ...
        'MaxIter', 1e6, 'RelTol', 1e-6, 'NumLambda', numlambda);
end
end

