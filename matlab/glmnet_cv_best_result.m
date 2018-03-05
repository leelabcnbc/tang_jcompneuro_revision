function [predicted_y, B_global, FitInfo_global, dev_sum] = glmnet_cv_best_result(X, y, ...
    standardize, family, foldid, alpha, debug_flag, lambda)
%GLMNET_CV_BEST_RESULT a wrapper for 1 call of lassoglm with CV
% and 5 calls without to get the raw test response.

if nargin < 8 || isempty(lambda)
    lambda = [];
end

if nargin < 7 || isempty(debug_flag)
    debug_flag = false;
end
    

[N, ~] = size(X);
assert(ismatrix(X));
assert(isequal(size(y), [N, 1]));
foldid = double(foldid(:));
assert(isequal(size(foldid), [N, 1]));
numfold = numel(unique(foldid));
assert(isequal(1:numfold, reshape(unique(foldid), [1, numfold])));

cvp = cvpartition(N, 'KFold', numfold);
assert(isequal(size(foldid), size(cvp.Impl.indices)));
% so other things are correct.
assert(isequal(sort(foldid), sort(cvp.Impl.indices)));
cvp.Impl.indices = foldid;

if debug_flag
    for i_fold = 1:numfold
        assert(isequal(cvp.test(i_fold), foldid==i_fold));
        assert(isequal(cvp.training(i_fold), foldid~=i_fold));
    end
end


if isequal(family, 'poisson')
    [B_global, FitInfo_global] = lassoglm_fix(X, y, 'poisson', 'Alpha', alpha, ...
        'Standardize', standardize, 'CV', cvp, 'Lambda', lambda, ...
        'MaxIter', 1e6, 'RelTol', 1e-6);
elseif isequal(family, 'gaussian')
    [B_global, FitInfo_global] = lassoglm_fix(X, y, 'normal', 'Alpha', alpha, ...
        'Standardize', standardize, 'CV', cvp, 'Lambda', lambda,...
        'MaxIter', 1e6, 'RelTol', 1e-6);
else
    assert(isequal(family, 'softplus'));
    [B_global, FitInfo_global] = lassoglm_fix(X, y, 'normal', 'Alpha', alpha, ...
        'Standardize', standardize, 'CV', cvp, 'Link', softplus_handles(), ...
        'Lambda', lambda, 'MaxIter', 1e6, 'RelTol', 1e-6);
end

% then use the smallest one to fetch some data.
lambda_best = FitInfo_global.Lambda(FitInfo_global.IndexMinDeviance);

predicted_y = zeros(N,1);
dev_sum = 0;

% make sure cv obj doesn't get screwed up.
if debug_flag
    for i_fold = 1:numfold
        assert(isequal(cvp.test(i_fold), foldid==i_fold));
        assert(isequal(cvp.training(i_fold), foldid~=i_fold));
    end
end

for fold_id = 1:numfold
    X_this = X(cvp.training(fold_id),:);
    y_this = y(cvp.training(fold_id),:);
    
    
    X_this_test = X(cvp.test(fold_id),:);
    y_this_test = y(cvp.test(fold_id),:);
    
    ratio_this = size(X,1)/size(X_this_test,1);
    
    % then fit.
    if isequal(family, 'poisson')
        [B, FitInfo] = lassoglm_fix(X_this, y_this, 'poisson', 'Alpha', alpha, ...
            'Standardize', standardize, 'Lambda', lambda_best, ...
            'MaxIter', 1e6, 'RelTol', 1e-6);
        y_predict = exp(X_this_test*B + FitInfo.Intercept);
        predicted_y(cvp.test(fold_id)) = y_predict;
        
        if debug_flag
            % http://thestatsgeek.com/2014/04/26/deviance-goodness-of-fit-test-for-poisson-regression/
            % it's actually a constant factor off from loglikelihood.
            dev_sum = dev_sum + sum(funcs.dev_poisson(y_predict, y_this_test))*ratio_this;
        end
        
    elseif isequal(family, 'gaussian')
        [B, FitInfo] = lassoglm_fix(X_this, y_this, 'normal', 'Alpha', alpha, ...
            'Standardize', standardize, 'Lambda', lambda_best, ...
            'MaxIter', 1e6, 'RelTol', 1e-6);
        y_predict = X_this_test*B + FitInfo.Intercept;
        predicted_y(cvp.test(fold_id)) = y_predict;
        if debug_flag
            dev_sum = dev_sum + sum((y_predict - y_this_test).^2)*ratio_this;
        end
    else
        assert(isequal(family, 'softplus'));
        [B, FitInfo] = lassoglm_fix(X_this, y_this, 'normal', 'Alpha', alpha, ...
            'Standardize', standardize, 'Link', softplus_handles(), ...
            'Lambda', lambda_best, ...
            'MaxIter', 1e6, 'RelTol', 1e-6);
        y_predict = funcs.softplus(X_this_test*B + FitInfo.Intercept);
        predicted_y(cvp.test(fold_id)) = y_predict;
        if debug_flag
            dev_sum = dev_sum + sum((y_predict - y_this_test).^2)*ratio_this;
        end
    end
    
    
end
dev_sum = dev_sum/numfold;
end

