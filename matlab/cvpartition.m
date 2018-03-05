classdef cvpartition
    %CVPARTITION Create a cross-validation partition for data.
    %   An object of the CVPARTITION class defines a random partition on a
    %   set of data of a specified size.  This partition can be used to
    %   define test and training sets for validating a statistical model
    %   using cross-validation.
    %
    %   C = CVPARTITION(N,'KFold',K) creates a CVPARTITION object C
    %   defining a random partition for K-fold cross-validation on N
    %   observations. The partition divides N observations into K disjoint
    %   subsamples (folds), chosen randomly but with roughly equal size.
    %   The default value of K is 10.
    %
    %   C = CVPARTITION(GROUP,'KFold',K) creates a CVPARTITION object C
    %   defining a random partition for a stratified K-fold
    %   cross-validation. GROUP is a vector indicating the class
    %   information for each observation. GROUP can be a categorical
    %   variable, a numeric vector, a string array, or a cell array of
    %   strings. Each subsample has roughly equal size and roughly the same
    %   class proportions as in GROUP. CVPARTITION treats NaNs or empty
    %   strings in GROUP as missing values.
    %
    %   C = CVPARTITION(N,'HoldOut',P) creates a CVPARTITION object C
    %   defining a random partition for hold-out validation on N
    %   observations. This partition divides N observations into a training
    %   set and a test set. P must be a scalar. When 0<P<1, CVPARTITION
    %   randomly selects approximately P*N observations for the test set.
    %   When P is an integer, it randomly selects P observations for the
    %   test set. The default value of P is 1/10.
    %
    %   C = CVPARTITION(GROUP,'HoldOut',P) randomly partitions observations
    %   into a training set and a test set with stratification, using the
    %   class information in GROUP, i.e., both training and test sets have
    %   roughly the same class proportions as in GROUP.
    %
    %   C = CVPARTITION(N,'LeaveOut') creates an object C defining a random
    %   partition for leave-one-out cross-validation on N observations.
    %   Leave-one-out is a special case of K-fold in which the number of
    %   folds is equal to the number of observations N.
    %
    %   C = CVPARTITION(N,'Resubstitution') creates a CVPARTITION object C
    %   which does not partition the data. Both the training set and the
    %   test set contain all of the original N observations.
    %
    %   CVPARTITION properties:
    %      Type               - Type of validation partition. 
    %      NumObservations    - Number of observations.
    %      NumTestSets        - Number of test sets. 
    %      TrainSize          - Size of each training set. 
    %      TestSize           - Size of each test set. 
    %
    %   CVPARTITION methods:
    %      repartition        - Rerandomize a cross-validation partition.
    %      test               - Test set for a cross-validation partition.
    %      training           - Training set for a cross-validation partition.
    %
    %   Example: Use 10-fold stratified cross-validation to compute the
    %   misclassification error for CLASSIFY on iris data.
    %
    %     load('fisheriris');
    %     CVO = cvpartition(species,'KFold',10);
    %     err = zeros(CVO.NumTestSets,1);
    %     for i = 1:CVO.NumTestSets
    %         trIdx = CVO.training(i);
    %         teIdx = CVO.test(i);
    %         ytest = classify(meas(teIdx,:),meas(trIdx,:),species(trIdx,:));
    %         err(i) = sum(~strcmp(ytest,species(teIdx)));
    %     end
    %     cvErr = sum(err)/sum(CVO.TestSize);
    %
    %   See also CVPARTITION/TEST, CVPARTITION/TRAINING, CVPARTITION/REPARTITION, CROSSVAL.
    
    %   Copyright 2007-2014 The MathWorks, Inc.


    properties(GetAccess = public, SetAccess = protected, Dependent=true)
       %TYPE  Validation partition type. 
       %   The Type property is a string with value  'kfold', 'holdout',
       %   'leaveout' or 'resubstitution'. It indicates the type of
       %   validation partition.
        Type; 
        
       %NUMTESTSETS Number of test sets.
       %   The NumTestSets property is an integer indicating the number of
       %   folds in K-fold and Leave-one-out. The value is one in Holdout
       %   and Resubstitution.
        NumTestSets;
        
        %TRAINSIZE Size of each training set.
        %   The TrainSize property indicates the size of each training set.
        %   It is a numeric vector in K-fold and Leave-one-out, or a
        %   numeric scalar in Holdout and Resubstitution.
        TrainSize;
        
        %TESTSIZE Size of each test set.
        %   The TestSize property indicates the size of each test set. It
        %   is a numeric vector in K-fold and Leave-one-out, or a numeric
        %   scalar in Holdout and Resubstitution.
        TestSize;
        
        %NUMOBSERVATIONS Number of observations.
        %   The NumObservations property is a numeric scalar holding the number of
        %   observations, including observations with missing GROUP values
        %   if GROUP is provided.
         NumObservations;        
    end
    
    properties(GetAccess = public, SetAccess = protected, Hidden=true, Dependent=true)
         N;    
    end

    properties(GetAccess = public, SetAccess = public, Hidden = true)
        Impl;
    end

    properties(GetAccess = private) 
        % Modify this property every time an incompatible change is
        % introduced into the cvpartition class so loadobj can intercept
        % when Matlab is trying to load incompatible instances.
        % Should be in sync with currentVersion variable used inside the
        % loadobj method.
        Version = 1; % R2016b
    end
    
    methods
        function n = get.N(this)
            n = this.Impl.N;
        end
        function n = get.Type(this)
            n = this.Impl.Type;
        end
        function n = get.NumTestSets(this)
            n = this.Impl.NumTestSets;
        end
        function n = get.TrainSize(this)
            n = this.Impl.TrainSize;
        end
        function n = get.TestSize(this)
            n = this.Impl.TestSize;
        end
        function n = get.NumObservations(this)
            n = this.Impl.NumObservations;
        end        
        function cv = cvpartition(varargin)
            disp('new CV');
            if nargin>0 && isstruct(varargin{1}) && isfield(varargin{1},'BckCmpBackdoorConstructor')
                error('cannot be here');
                % Backdoor constructor used only by cvpartition.loadobj for
                % compatibility compliance.
                cv.Impl = internal.stats.cvpartitionInMemoryImpl(varargin{1});
            elseif (nargin>0 && iscell(varargin{1}) && istall(varargin{1}{1}))
                error('cannot be here');
                % Backdoor constructor for tall variables used only by
                % tall.cvpartition by wraping a tall within a cell and
                % calling this constructor.
                cv.Impl = internal.stats.bigdata.cvpartitionTallImpl(varargin{:});
            else
                % Regular constructor
                cv.Impl = cvpartitionInMemoryImpl(varargin{:});
            end
        end % cvpartition constructor

        function  cv = repartition(cv,varargin)
        %REPARTITION Rerandomize a cross-validation partition. 
        %   D = REPARTITION(C) creates a new random cross-validation partition D
        %   of the same type and size as C.  Use REPARTITION to perform multiple
        %   Monte-Carlo repetitions of cross-validation.
        %   D = REPARTITION(C,S) uses the RandStream object S as its
        %   random number generator.
        %   
        %   See also CVPARTITION.
        cv.Impl = repartition(cv.Impl,varargin{:});
        end % repartition
       
        function trainIndices = training(cv,varargin)
        %TRAINING Training set for a cross-validation partition.
        %   TRIDX = TRAINING(C) returns a logical vector TRIDX that selects
        %   the observations in the training set for the hold-out
        %   cross-validation partition C.  C may also be a partition for
        %   resubstitution, in which case TRIDX is a logical vector that
        %   selects all observations.
        %
        %   TRIDX = TRAINING(C,I) returns a logical vector TRIDX that selects
        %   the observations in the I-th training set for a K-fold or
        %   leave-one-out cross-validation partition C.  In K-fold
        %   cross-validation, C divides a data set into K disjoint folds with
        %   roughly equal size.  The I-th training set consists of all
        %   observations not contained in the I-th fold.  In leave-one-out
        %   cross-validation, the I-th training set consists of the entire
        %   data set except the I-th observation.
        %
        %   See also CVPARTITION, CVPARTITION/TEST.
        trainIndices = training(cv.Impl,varargin{:});
        end

        function testIndices = test(cv,varargin)
        %TEST Test set for a cross-validation partition.
        %   TEIDX = TEST(C) returns a logical vector TEIDX that selects the
        %   observations in the test set for the hold-out cross-validation
        %   partition C.  C may also be a partition for resubstitution, in
        %   which case TEIDX is a logical vector that selects all
        %   observations.
        %
        %   TEIDX = TEST(C,I) returns a logical vector TEIDX that selects the
        %   observations in the I-th test set for a K-fold or leave-one-out
        %   cross-validation partition C.  In K-fold cross-validation, C
        %   divides a data set into K disjoint folds with roughly equal size.
        %   The I-th test set consists of the I-th fold.  In leave-one-out
        %   cross-validation, the I-th test set consists of the I-th
        %   observation.
        %
        %   See also CVPARTITION, CVPARTITION/TRAINING.
        testIndices = test(cv.Impl,varargin{:});
        end

        % Display methods
%         function display(cv)
%             objectname = inputname(1);
%             display(cv.Impl,objectname)
%         end
        function disp(cv)
            disp(cv.Impl)
        end
    end % public methods block

    
    methods(Hidden = true)
        function b = fieldnames(cv)
            b = fieldnames(cv.Impl);
        end
        
        % Methods that we inherit, but do not want
        function a = fields(varargin),     throwUndefinedError(); end
        function a = ctranspose(varargin), throwUndefinedError(); end
        function a = transpose(varargin),  throwUndefinedError(); end
        function a = permute(varargin),    throwUndefinedError(); end
        function a = reshape(varargin),    throwUndefinedError(); end
        function a = cat(varargin),        throwNoCatError(); end
        function a = horzcat(varargin),    throwNoCatError(); end
        function a = vertcat(varargin),    throwNoCatError(); end
    end
    methods(Hidden = true, Static = true)
        function a = empty(varargin)
            error(message('stats:cvpartition:NoEmptyAllowed', upper( mfilename )));
        end
    end
    
    methods(Hidden, Static, Access='public')
         function cv = loadobj(cv)
             currentVersion = 1; % Should be in sync with the cv.Version property
             if isstruct(cv)
                 % The incompatibility introduced in 16b will always reach
                 % this point with a struct (not a object). cvpartition
                 % constructor has a backdoor for this case, so we do not
                 % need to take any other preventive action now.
                 cv.BckCmpBackdoorConstructor = true;
                 cv = cvpartition(cv);
             elseif isempty(cv.Version) || (cv.Version < currentVersion)
                 % This branch should never be reached
                 error(message('stats:cvpartition:FwrdCompatibility'))
             end
         end
    end % hidden static public
   
end % classdef

function throwNoCatError()
error(message('stats:cvpartition:NoCatAllowed', upper( mfilename )));
end

function throwUndefinedError()
st = dbstack;
name = regexp(st(2).name,'\.','split');
error(message('stats:cvpartition:UndefinedFunction', name{ 2 }, mfilename));
end
