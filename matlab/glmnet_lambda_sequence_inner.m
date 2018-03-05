function [lambda] = glmnet_lambda_sequence_inner(x,y,distr,varargin)
%LASSOGLM Perform lasso or elastic net regularization for a generalized linear model.
%   [B,STATS] = LASSOGLM(X,Y,DISTR,...) Performs L1-penalized maximum likelihood 
%   fits (lasso) relating the predictors in X to the responses in Y, or 
%   fits subject to joint L1- and L2 constraints (elastic net).
%   The default is a lasso-style fit, that is, a maximum likelihood fit
%   subject to a constraint on the L1-norm of the coefficients B.
%
%   LASSOGLM accepts all the command line parameters of the LASSO function 
%   and it accepts command line parameters of the GLMFIT function, with 
%   the following exceptions. LASSOGLM also accepts the arguments 'Link'
%   and 'Offset' of GLMFIT (they can also be lower case 'link' or 'offset').  
%   LASSOGLM It does not accept the argument 'estdisp' of GLMFIT, or the 
%   argument 'constant'.  LASSOGLM does not calculate standard errors
%   or covariances among the coefficients, as GLMFIT does.
%   LASSOGLM always calculates an Intercept term. 
%   
%   Positional parameters:
%
%     X                A numeric matrix (dimension, say, NxP)
%     Y                A numeric vector of length N.  If DISTR is
%                      'binomial', Y may be a binary vector indicating
%                      success/failure, and the total number of trials is
%                      taken to be 1 for all observations.  If DISTR is
%                      'binomial', Y may also be a two column matrix, the 
%                      first column containing the number of successes for
%                      each observation, and the second column containing
%                      the total number of trials.
%     DISTR            The distributional family for the non-systematic
%                      variation in the responses.  Acceptable values for DISTR are
%                      'normal', 'binomial', 'poisson', 'gamma', and 'inverse gaussian'.  
%                      By default, the distribution is fit using the canonical link 
%                      corresponding to DISTR.  Other link functions may be
%                      specified using the optional parameter 'link'.
%   
%   Optional input parameters:  
%
%     'Weights'        Observation weights.  Must be a vector of non-negative
%                      values, of the same length as columns of X.  At least
%                      two values must be positive. (default ones(N,1) or 
%                      equivalently (1/N)*ones(N,1)).
%     'Alpha'          Elastic net mixing value, or the relative balance
%                      between L2 and L1 penalty (default 1, range (0,1]).
%                      Alpha=1 ==> lasso, otherwise elastic net.
%                      Alpha near zero ==> nearly ridge regression.
%     'NumLambda'      The number of lambda values to use, if the parameter
%                      'Lambda' is not supplied (default 100). Ignored
%                      if 'Lambda' is supplied. LASSOGLM may return fewer
%                      fits than specified by 'NumLambda' if the deviance
%                      of the fits drops below a threshold percentage of
%                      the null deviance (deviance of the fit without
%                      any predictors X).
%     'LambdaRatio'    Ratio between the minimum value and maximum value of
%                      lambda to generate, if the  parameter "Lambda" is not 
%                      supplied.  Legal range is [0,1). Default is 0.0001.
%                      If 'LambdaRatio' is zero, LASSOGLM will generate its
%                      default sequence of lambda values but replace the
%                      smallest value in this sequence with the value zero.
%                      'LambdaRatio' is ignored if 'Lambda' is supplied.
%     'Lambda'         Lambda values. Will be returned in return argument
%                      STATS in ascending order. The default is to have
%                      LASSOGLM generate a sequence of lambda values, based 
%                      on 'NumLambda' and 'LambdaRatio'. LASSOGLM will generate 
%                      a sequence, based on the values in X and Y, such that 
%                      the largest Lambda value is estimated to be just 
%                      sufficient to produce all zero coefficients B. 
%                      You may supply a vector of real, non-negative values 
%                      of lambda for LASSOGLM to use, in place of its default
%                      sequence.  If you supply a value for 'Lambda', 
%                      'NumLambda' and 'LambdaRatio' are ignored.
%     'DFmax'          Maximum number of non-zero coefficients in the model.
%                      Can be useful with large numbers of predictors.
%                      Results only for lambda values that satisfy this
%                      degree of sparseness will be returned.  Default is
%                      to not limit the number of non-zero coefficients.
%     'Standardize'    Whether to scale X prior to fitting the model
%                      sequence. This affects whether the regularization is
%                      applied to the coefficients on the standardized
%                      scale or the original scale. The results are always
%                      presented on the original data scale. Default is
%                      TRUE, do scale X.
%     'RelTol'         Convergence threshold for coordinate descent algorithm.
%                      The coordinate descent iterations will terminate
%                      when the relative change in the size of the
%                      estimated coefficients B drops below this threshold.
%                      Default: 1e-4. Legal range is (0,1).
%     'CV'             If present, indicates the method used to compute Deviance.
%                      When 'CV' is a positive integer K, LASSOGLM uses K-fold
%                      cross-validation.  Set 'CV' to a cross-validation 
%                      partition, created using CVPARTITION, to use other
%                      forms of cross-validation. You cannot use a
%                      'Leaveout' partition with LASSOGLM.                
%                      When 'CV' is 'resubstitution', LASSOGLM uses X and Y 
%                      both to fit the model and to estimate the deviance 
%                      of the fitted model, without cross-validation.  
%                      The default is 'resubstitution'.
%     'MCReps'         A positive integer indicating the number of Monte-Carlo
%                      repetitions for cross-validation.  The default value is 1.
%                      If 'CV' is 'resubstitution' or a cvpartition of type
%                      'resubstitution', 'MCReps' must be 1.  If 'CV' is a
%                      cvpartition of type 'holdout', then 'MCReps' must be
%                      greater than one.
%     'MaxIter'        Maximum number of iterations allowed.  Default is 1e4.
%     'PredictorNames' A cell array of names for the predictor variables,
%                      in the order in which they appear in X. 
%                      Default: {}
%     'Options'        A structure that contains options specifying whether to
%                      conduct cross-validation evaluations in parallel, and
%                      options specifying how to use random numbers when computing
%                      cross validation partitions. This argument can be created
%                      by a call to STATSET. CROSSVAL uses the following fields:
%                        'UseParallel'
%                        'UseSubstreams'
%                        'Streams'
%                      For information on these fields see PARALLELSTATS.
%                      NOTE: If supplied, 'Streams' must be of length one.
%     'Link'           The link function to use in place of the canonical link.
%                      The link function defines the relationship f(mu) = x*b
%                      between the mean response mu and the linear combination of
%                      predictors x*b.  Specify the link parameter value as one of
%                      - the text strings 'identity', 'log', 'logit', 'probit',
%                        'comploglog', 'reciprocal', 'loglog', or
%                      - an exponent P defining the power link, mu = (x*b)^P 
%                        for x*b > 0, or
%                      - a cell array of the form {FL FD FI}, containing three
%                        function handles, created using @, that define the link (FL),
%                        the derivative of the link (FD), and the inverse link (FI).  
%     'Offset'         A vector to use as an additional predictor variable, but
%                      with a coefficient value fixed at 1.0.  Default is
%                      not to utilize an offset variable.
% 
%   Return values:
%     B                The fitted coefficients for each model. 
%                      B will have dimension PxL, where 
%                      P = size(X,2) is the number of predictors, and
%                      L = length(STATS.Lambda).
%     STATS            STATS is a struct that contains information about the
%                      sequence of model fits corresponding to the columns
%                      of B. STATS contains the following fields:
%
%       'Intercept'    The intercept term for each model. Dimension 1xL.
%       'Lambda'       The sequence of lambda penalties used, in ascending order. 
%                      Dimension 1xL.
%       'Alpha'        The elastic net mixing value that was used.
%       'DF'           The number of nonzero coefficients in B for each
%                      value of lambda. Dimension 1xL.
%       'Deviance'     The deviance of the fitted model for each value of
%                      lambda. If cross-validation was performed, the values 
%                      for 'Deviance' represent the estimated expected 
%                      deviance of the model applied to new data, as 
%                      calculated by cross-validation. Otherwise, 
%                      'Deviance' is the deviance of the fitted model 
%                      applied to the data used to perform the fit. 
%                      Dimension 1xL.
%
%     If cross-validation was performed, STATS also includes the following
%     fields:
%
%       'SE'                The standard error of 'Deviance' for each lambda, as
%                           calculated during cross-validation. Dimension 1xL.
%       'LambdaMinDeviance' The lambda value with minimum expected deviance, as 
%                           calculated during cross-validation. Scalar.
%       'Lambda1SE'         The largest lambda such that 'Deviance' is within 
%                           one standard error of the minimum. Scalar.
%       'IndexMinDeviance'  The index of Lambda with value LambdaMinMSE. Scalar.
%       'Index1SE'          The index of Lambda with value Lambda1SE. Scalar.
%
%   See also lassoPlot, lasso, ridge, parallelstats, glmfit.

%   References: 
%   [1] Tibshirani, R. (1996) Regression shrinkage and selection
%       via the lasso. Journal of the Royal Statistical Society,
%       Series B, Vol 58, No. 1, pp. 267-288.
%   [2] Zou, H. and T. Hastie. (2005) Regularization and variable
%       selection via the elastic net. Journal of the Royal Statistical
%       Society, Series B, Vol. 67, No. 2, pp. 301-320.
%   [3] Friedman, J., R. Tibshirani, and T. Hastie. (2010) Regularization
%       paths for generalized linear models via coordinate descent.
%       Journal of Statistical Software, Vol 33, No. 1,
%       http://www.jstatsoft.org/v33/i01.
%   [4] Hastie, T., R. Tibshirani, and J. Friedman. (2008) The Elements
%       of Statistical Learning, 2nd edition, Springer, New York.
%   [5] Dobson, A.J. (2002) An Introduction to Generalized Linear
%       Models, 2nd edition, Chapman&Hall/CRC Press.
%   [6] McCullagh, P., and J.A. Nelder (1989) Generalized Linear
%       Models, 2nd edition, Chapman&Hall/CRC Press.
%   [7] Collett, D. (2003) Modelling Binary Data, 2nd edition,
%       Chapman&Hall/CRC Press.

%   Copyright 2011-2016 The MathWorks, Inc.


if nargin < 2
    error(message('stats:lassoGlm:TooFewInputs'));
end

if nargin < 3 || isempty(distr), distr = 'normal'; end

paramNames = {     'link' 'offset' 'weights'};
paramDflts = {'canonical'  []       []};
[link,offset,pwts,~,varargin] = ...
                    internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
                
% Read in the optional parameters pertinent to regularization (eg, lasso)
LRdefault = 1e-4;
pnames = { 'alpha' 'numlambda' 'lambdaratio' 'lambda' ...
    'dfmax' 'standardize' 'reltol' 'cv' 'mcreps' 'maxiter' ...
    'predictornames' 'options' };
dflts  = {  1       100       LRdefault     []      ...
     []      true          1e-4    'resubstitution'  1  1e4 ...
     {}               []};
[alpha, nLambda, lambdaRatio, lambda, ...
    dfmax, standardize, reltol, cvp, mcreps, maxIter, predictorNames, parallelOptions] ...
     = internal.stats.parseArgs(pnames, dflts, varargin{:});
 
if ~isempty(lambda)
    userSuppliedLambda = true;
else
    userSuppliedLambda = false;
end

% X a real 2D matrix
if ~ismatrix(x) || length(size(x)) ~= 2 || ~isreal(x)
    error(message('stats:lassoGlm:XnotaReal2DMatrix'));
end

% We need at least two observations.
if isempty(x) || size(x,1) < 2
    error(message('stats:lassoGlm:TooFewObservations'));
end

% Categorical responses 'binomial'
if isa(y,'categorical')
    [y, classname] = grp2idx(y); 
    nc = length(classname);
    if nc > 2
        error(message('stats:glmfit:TwoLevelCategory'));
    end
    y(y==1) = 0;
    y(y==2) = 1;
end

% Number of Predictors
P = size(x,2);

% Head off potential cruft in the command window.
wsIllConditioned2 = warning('off','stats:glmfit:IllConditioned');
cleanupIllConditioned2 = onCleanup(@() warning(wsIllConditioned2));

% Sanity checking on predictors, responses, weights and offset parameter,
% and removal of NaNs and Infs from same. Also, conversion of the
% two-column form of binomial response to a proportion.
[X, Y, offset, pwts, dataClass, nTrials, binomialTwoColumn] = ...
    glmProcessData(x, y, distr, 'off', offset, pwts);

[~,sqrtvarFun,devFun,linkFun,dlinkFun,ilinkFun,link,mu,eta,muLims,isCanonical,dlinkFunCanonical] = ...
    glmProcessDistrAndLink(Y,distr,link,'off',nTrials,dataClass);

[X,Y,pwts,nLambda,lambda,dfmax,cvp,mcreps,predictorNames,ever_active] = ...
    processLassoParameters(X,Y,pwts,alpha,nLambda,lambdaRatio,lambda,dfmax, ...
    standardize,reltol,cvp,mcreps,predictorNames);

% Compute the amount of penalty at which all coefficients shrink to zero.
[lambdaMax, nullDev, nullIntercept] = computeLambdaMax(X, Y, pwts, alpha, standardize, ...
    distr, link, dlinkFun, offset, isCanonical, dlinkFunCanonical, devFun);

% If the command line did not provide a sequence of penalty terms,
% generate a sequence.
if isempty(lambda)
    lambda = computeLambdaSequence(lambdaMax, nLambda, lambdaRatio, LRdefault);
end

nLambda = length(lambda);
reverseIndices = nLambda:-1:1;
lambda = lambda(reverseIndices);
lambda = reshape(lambda,1,nLambda);

end %-main block

% ------------------------------------------
% SUBFUNCTIONS 
% ------------------------------------------

% ===================================================
%                  startingVals() 
% ===================================================

function mu = startingVals(distr,y,N)
% Find a starting value for the mean, avoiding boundary values
switch distr
case 'poisson'
    mu = y + 0.25;
case 'binomial'
    mu = (N .* y + 0.5) ./ (N + 1);
case {'gamma' 'inverse gaussian'}
    mu = max(y, eps(class(y))); % somewhat arbitrary
otherwise
    mu = y;
end
end %-startingVals


% ===============================================
%               glmProcessData() 
% ===============================================

function [x, y, offset, pwts, dataClass, N, binomialTwoColumn] = ...
    glmProcessData(x, y, distr, const, offset, pwts)

N = []; % needed only for binomial
binomialTwoColumn = false;

% Convert the two-column form of 'y', if supplied ('binomial' only).
if strcmp(distr,'binomial')
    if size(y,2) == 1
        % N will get set to 1 below
        if any(y < 0 | y > 1)
            error(message('stats:lassoGlm:BadDataBinomialFormat'));
        end
    elseif size(y,2) == 2
        binomialTwoColumn = true;
        y(y(:,2)==0,2) = NaN;
        N = y(:,2);
        y = y(:,1) ./ N;
        if any(y < 0 | y > 1)
            error(message('stats:lassoGlm:BadDataBinomialRange'));
        end
    else
        error(message('stats:lassoGlm:MatrixOrBernoulliRequired'));
    end
end

[anybad,~,y,x,offset,pwts,N] = dfswitchyard('statremovenan',y,x,offset,pwts,N);
if anybad > 0
    switch anybad
    case 2
        error(message('stats:lassoGlm:InputSizeMismatchX'))
    case 3
        error(message('stats:lassoGlm:InputSizeMismatchOffset'))
    case 4
        error(message('stats:lassoGlm:InputSizeMismatchPWTS'))
    end
end

% Extra screening for lassoglm (Infs and zero weights)
okrows = all(isfinite(x),2) & all(isfinite(y),2) & all(isfinite(offset));

if ~isempty(pwts)
    % This screen works on weights prior to stripping NaNs and Infs.
    if ~isvector(pwts) || ~isreal(pwts) || size(x,1) ~= length(pwts) || ...
            ~all(isfinite(pwts)) || any(pwts<0)
        error(message('stats:lassoGlm:InvalidObservationWeights'));
    end    
    okrows = okrows & pwts(:)>0;
    pwts = pwts(okrows);
end

% We need at least two observations after stripping NaNs and Infs and zero weights.
if sum(okrows)<2
    error(message('stats:lassoGlm:TooFewObservationsAfterNaNs'));
end

% Remove observations with Infs in the predictor or response
% or with zero observation weight.  NaNs were already gone.
x = x(okrows,:);
y = y(okrows);
if ~isempty(N) && ~isscalar(N)
    N = N(okrows);
end
if ~isempty(offset)
    offset = offset(okrows);
end

if isequal(const,'on')
    x = [ones(size(x,1),1) x];
end
dataClass = superiorfloat(x,y);
x = cast(x,dataClass);
y = cast(y,dataClass);

if isempty(offset), offset = 0; end
if isempty(N), N = 1; end

end %-glmProcessData()

% ===================================================
%             processLassoParameters() 
% ===================================================

function [X,Y,weights,nLambda,lambda,dfmax,cvp,mcreps,predictorNames,ever_active] = ...
    processLassoParameters(X,Y,weights, alpha, nLambda, lambdaRatio, lambda, dfmax, ...
    standardize, reltol, cvp, mcreps, predictorNames)

% === 'Weights' parameter ===
if ~isempty(weights)
        
    % Computations expect that weights is a row vector.
    weights = weights(:)';
    
end

[~,P] = size(X);

% If X has any constant columns, we want to exclude them from the
% coordinate descent calculations.  The corresponding coefficients
% will be returned as zero.
constantPredictors = (range(X)==0);
ever_active = ~constantPredictors;

% === 'Alpha' parameter ===

% Require 0 < alpha <= 1.
% "0" would correspond to ridge, "1" is lasso.
if ~isscalar(alpha) || ~isreal(alpha) || ~isfinite(alpha) || ...
        alpha <= 0 || alpha > 1
    error(message('stats:lassoGlm:InvalidAlpha'))
end

% === 'Standardize' option ===

% Require a logical value.
if ~isscalar(standardize) || (~islogical(standardize) && standardize~=0 && standardize~=1)
    error(message('stats:lassoGlm:InvalidStandardize'))
end

% === 'Lambda' sequence or 'NumLambda' and 'lambdaRatio' ===

if ~isempty(lambda)
    
    % Sanity check on user-supplied lambda sequence.  Should be non-neg real.
    if ~isreal(lambda) || any(lambda < 0)
        error(message('stats:lassoGlm:NegativeLambda'));
    end

    lambda = sort(lambda(:),1,'descend');
    
else
    
    % Sanity-check of 'NumLambda', should be positive integer.
    if ~isreal(nLambda) || ~isfinite(nLambda) || nLambda < 1
        error(message('stats:lassoGlm:InvalidNumLambda'));
    else
        nLambda = floor(nLambda);
    end
    
    % Sanity-checking of LambdaRatio, should be in [0,1).
    if ~isreal(lambdaRatio) || lambdaRatio <0 || lambdaRatio >= 1
        error(message('stats:lassoGlm:InvalidLambdaRatio'));
    end
end

% === 'RelTol' parameter ===
%
if ~isscalar(reltol) || ~isreal(reltol) || ~isfinite(reltol) || reltol <= 0 || reltol >= 1
    error(message('stats:lassoGlm:InvalidRelTol'));
end

% === 'DFmax' parameter ===
%
% DFmax is #non-zero coefficients 
% DFmax should map to an integer in [1,P] but we truncate if .gt. P
%
if isempty(dfmax)
    dfmax = P;
else
    if ~isscalar(dfmax)
        error(message('stats:lassoGlm:DFmaxBadType'));
    end
    try
        dfmax = uint32(dfmax);
    catch ME
        mm = message('stats:lassoGlm:DFmaxBadType');
        throwAsCaller(MException(mm.Identifier,'%s',getString(mm)));
    end
    if dfmax < 1
        error(message('stats:lassoGlm:DFmaxNotAnIndex'));
    else
        dfmax = min(dfmax,P);
    end
end

% === 'Mcreps' parameter ===
%
if ~isscalar(mcreps) || ~isreal(mcreps) || ~isfinite(mcreps) || mcreps < 1
    error(message('stats:lassoGlm:MCRepsBadType'));
end
mcreps = fix(mcreps);

% === 'CV' parameter ===
%

if isnumeric(cvp) && isscalar(cvp) && (cvp==round(cvp)) && (0<cvp)
    % cvp is a kfold value. It will be passed as such to crossval.
    if (cvp>size(X,1))
        error(message('stats:lassoGlm:InvalidCVforX'));
    end
    cvp = cvpartition(size(X,1),'Kfold',cvp);
elseif isa(cvp,'cvpartition')
    if strcmpi(cvp.Type,'resubstitution')
        cvp = 'resubstitution';
    elseif strcmpi(cvp.Type,'leaveout')
        error(message('stats:lassoGlm:InvalidCVtype'));
    elseif strcmpi(cvp.Type,'holdout') && mcreps<=1
        error(message('stats:lassoGlm:InvalidMCReps'));
    end
elseif strncmpi(cvp,'resubstitution',length(cvp))
    % This may have been set as the default, or may have been
    % provided at the command line.  In case it's the latter, we
    % expand abbreviations.
    cvp = 'resubstitution';
else
    error(message('stats:lassoGlm:InvalidCVtype'));
end
if strcmp(cvp,'resubstitution') && mcreps ~= 1
    error(message('stats:lassoGlm:InvalidMCReps'));
end

if isa(cvp,'cvpartition')
    if (cvp.N ~= size(X,1)) || (min(cvp.TrainSize) < 2)
        % We need partitions that match the total number of observations
        % (after stripping NaNs and Infs and zero observation weights), and
        % we need training sets with at least 2 usable observations.
        error(message('stats:lassoGlm:TooFewObservationsForCrossval'));
    end
end

% === 'PredictorNames' parameter ===
%
% If PredictorNames is not supplied or is supplied as empty, we just 
% leave it that way. Otherwise, confirm that it is a cell array of strings.
%
if ~isempty(predictorNames) 
    if ~iscellstr(predictorNames) || length(predictorNames(:)) ~= size(X,2)
        error(message('stats:lassoGlm:InvalidPredictorNames'));
    else
        predictorNames = predictorNames(:)';
    end
end

end %-processLassoParameters()

% ===================================================
%             glmProcessDistrAndLink()
% ===================================================

function [estdisp,sqrtvarFun,devFun,linkFun,dlinkFun,ilinkFun,link,mu,eta,muLims,...
    isCanonical,dlinkFunCanonical] = ...
    glmProcessDistrAndLink(y,distr,link,estdisp,N,dataClass)

switch distr
    case 'normal'
        canonicalLink = 'identity';
    case 'binomial'
        canonicalLink = 'logit';
    case 'poisson'
        canonicalLink = 'log';
    case 'gamma'
        canonicalLink = 'reciprocal';
    case 'inverse gaussian'
        canonicalLink = -2;
end

if isequal(link, 'canonical'), link = canonicalLink; end

switch distr
case 'normal'
    sqrtvarFun = @(mu) ones(size(mu));
    devFun = @(mu,y) (y - mu).^2;
    estdisp = 'on';
case 'binomial'
    sqrtN = sqrt(N);
    sqrtvarFun = @(mu) sqrt(mu).*sqrt(1-mu) ./ sqrtN;
    devFun = @(mu,y) 2*N.*(y.*log((y+(y==0))./mu) + (1-y).*log((1-y+(y==1))./(1-mu)));
case 'poisson'
    if any(y < 0)
        error(message('stats:lassoGlm:BadDataPoisson'));
    end
    sqrtvarFun = @(mu) sqrt(mu);
    devFun = @(mu,y) 2*(y .* (log((y+(y==0)) ./ mu)) - (y - mu));
case 'gamma'
    if any(y <= 0)
        error(message('stats:lassoGlm:BadDataGamma'));
    end
    sqrtvarFun = @(mu) mu;
    devFun = @(mu,y) 2*(-log(y ./ mu) + (y - mu) ./ mu);
    estdisp = 'on';
case 'inverse gaussian'
    if any(y <= 0)
        error(message('stats:lassoGlm:BadDataInvGamma'));
    end
    sqrtvarFun = @(mu) mu.^(3/2);
    devFun = @(mu,y) ((y - mu)./mu).^2 ./  y;
    estdisp = 'on';
otherwise
    error(message('stats:lassoGlm:BadDistribution'));
end

% Instantiate functions for one of the canned links, or validate a
% user-defined link specification.
[linkFun,dlinkFun,ilinkFun] = dfswitchyard('stattestlink',link,dataClass);

% Initialize mu and eta from y.
mu = startingVals(distr,y,N);
eta = linkFun(mu);

% Enforce limits on mu to guard against an inverse link that doesn't map into
% the support of the distribution.
switch distr
case 'binomial'
    % mu is a probability, so order one is the natural scale, and eps is a
    % reasonable lower limit on that scale (plus it's symmetric).
    muLims = [eps(dataClass) 1-eps(dataClass)];
case {'poisson' 'gamma' 'inverse gaussian'}
    % Here we don't know the natural scale for mu, so make the lower limit
    % small.  This choice keeps mu^4 from underflowing.  No upper limit.
    muLims = realmin(dataClass).^.25;
otherwise
    muLims = []; 
end

if isequal(linkFun,@funcs.inv_softplus)
    % hack for this one
    muLims = realmin(dataClass).^.25;
end

% These two quantities (isCanonical, dlinkFunCanonical) are not used by 
% glmfit but they are needed for calculation of lambdaMax in lassoglm.

isCanonical = isequal(link, canonicalLink);
[~, dlinkFunCanonical] = dfswitchyard('stattestlink', canonicalLink, dataClass);

end %-glmProcessDistrAndLink()

% ===================================================
%                      glmIRLS()
% ===================================================


function [lambdaMax, nullDev, nullIntercept] = computeLambdaMax(X, Y, weights, alpha, standardize, ...
    distr, link, dlinkFun, offset, isCanonical, dlinkFunCanonical, devFun)

% lambdaMax is the penalty term (lambda) beyond which coefficients
% are guaranteed to be all zero: there is no benefit to calculating
% penalized fits with lambda > lambdaMax.
%
% nullDev is the deviance of the fit using just a constant term.
%
% The input parameter 'devFun' is used only as a sanity check to see if glmfit 
% gets a plausible fit with intercept term only.

% Head off potential cruft in the command window.
wsIllConditioned2 = warning('off','stats:glmfit:IllConditioned');
wsIterationLimit = warning('off','stats:glmfit:IterationLimit');
wsPerfectSeparation = warning('off','stats:glmfit:PerfectSeparation');
wsBadScaling = warning('off','stats:glmfit:BadScaling');
cleanupIllConditioned2 = onCleanup(@() warning(wsIllConditioned2));
cleanupIterationLimit = onCleanup(@() warning(wsIterationLimit));
cleanupPerfectSeparation = onCleanup(@() warning(wsPerfectSeparation));
cleanupBadScaling = onCleanup(@() warning(wsBadScaling));

if ~isempty(weights)
    observationWeights = true;
    weights = weights(:)';        
    % Normalized weights are used for standardization and calculating lambdaMax.
    normalizedweights = weights / sum(weights);
else
    observationWeights = false;
end

[N,~] = size(X);

% If we were asked to standardize the predictors, do so here because
% the calculation of lambdaMax needs the predictors as we will use
% them to perform fits.

if standardize
    % If X has any constant columns, we want to protect against
    % divide-by-zero in normalizing variances.
    constantPredictors = (range(X)==0);

    if ~observationWeights
        % Center and scale
        [X0,~,~] = zscore(X,1);
    else
        % Weighted center and scale
        muX = normalizedweights * X;
        X0 = bsxfun(@minus,X,muX);
        sigmaX = sqrt( normalizedweights * (X0.^2) );
        % Avoid divide by zero with constant predictors
        sigmaX(constantPredictors) = 1;
        X0 = bsxfun(@rdivide, X0, sigmaX);
    end
else
    if ~observationWeights
        % Center
        muX = mean(X,1);
        X0 = bsxfun(@minus,X,muX);
    else
        % Weighted center
        muX = normalizedweights(:)' * X;
        X0 = bsxfun(@minus,X,muX);
    end
end

constantTerm = ones(length(Y),1);
if isscalar(offset)
    [coeffs,nullDev] = glmfit(constantTerm,Y,distr,'constant','off', ...
        'link',link, 'weights',weights);
    predictedMu = glmval(coeffs,constantTerm,link,'constant','off');
else
    [coeffs,nullDev] = glmfit(constantTerm,Y,distr,'constant','off', ...
        'link',link,'weights',weights,'offset',offset);
    predictedMu = glmval(coeffs,constantTerm,link,'constant','off','offset',offset);
end

nullIntercept = coeffs;

% Sanity check. With badly matched link / distr / data, glmfit may not 
% have been able to converge to a reasonble estimate.  If so, the inputs
% to lassoglm may constitute a difficult problem formulation with 
% unsatisfactory maximum likelihood solution.  Poor formulations
% have been observed with mismatched links (ie, 'reciprocal' link with
% the Poisson distribution, in place of canonical 'log').  As a screen for
% this contingency, calculate the deviance we would get using the scalar
% unmodeled mean for the response data. Call this quantity "muDev".  
% The value of muDev should be no better than the nullDev calculated
% by glmfit above (albeit it might be equal or nearly equal).
% If the value is better, warn that the computations are of uncertain validity.

if observationWeights
    muDev = weights * devFun(mean(Y)*ones(length(Y),1), Y);
else
    muDev = sum(devFun(mean(Y)*ones(length(Y),1), Y));
end
if (muDev - nullDev) / max([1.0 muDev nullDev]) < - 1.0e-4
    [~, lastid] = lastwarn;
    if strcmp(lastid,'stats:glmfit:BadScaling')
        % This reassignment of predicted values is not guaranteed to
        % improve matters, but is an attempt to generate a workable
        % sequence of lambda values. Note: Since it is a constant
        % value for all observations, observation weights are a wash.
        predictedMu = mean(Y)*ones(length(Y),1);
        warning(message('stats:lassoGlm:DifficultLikelihood'));
    end
end

if ~isCanonical
    X0 = bsxfun( @times, X0, dlinkFunCanonical(predictedMu) ./ dlinkFun(predictedMu) );
end

if ~observationWeights
    dotp = abs(X0' * (Y - predictedMu));
    lambdaMax = max(dotp) / (N*alpha);
else
    wX0 = bsxfun(@times, X0, normalizedweights');
    dotp = abs(sum(bsxfun(@times, wX0, (Y - predictedMu))));
    lambdaMax = max(dotp) / alpha;
end

end %-computeLambdaMax()

function lambda = computeLambdaSequence(lambdaMax, nLambda, lambdaRatio, LRdefault)

% Fill in the log-spaced sequence of lambda values.
        
if nLambda==1
    lambda = lambdaMax;
else
    % Fill in a number "nLambda" of smaller values, on a log scale.
    if lambdaRatio==0
        lambdaRatio = LRdefault;
        addZeroLambda = true;
    else
        addZeroLambda = false;
    end
    lambdaMin = lambdaMax * lambdaRatio;
    loghi = log(lambdaMax);
    loglo = log(lambdaMin);
    lambda = exp(linspace(loghi,loglo,nLambda));
    if addZeroLambda
        lambda(end) = 0;
    else
        lambda(end) = lambdaMin;
    end
end

end %-computeLambdaSequence

