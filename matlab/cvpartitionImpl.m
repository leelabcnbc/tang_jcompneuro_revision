classdef cvpartitionImpl
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


    properties(GetAccess = public, SetAccess = protected)
       %TYPE  Validation partition type. 
       %   The Type property is a string with value  'kfold', 'holdout',
       %   'leaveout' or 'resubstitution'. It indicates the type of
       %   validation partition.
        Type = '';
        
       %NUMTESTSETS Number of test sets.
       %   The NumTestSets property is an integer indicating the number of
       %   folds in K-fold and Leave-one-out. The value is one in Holdout
       %   and Resubstitution.
        NumTestSets = [];
        
        %TRAINSIZE Size of each training set.
        %   The TrainSize property indicates the size of each training set.
        %   It is a numeric vector in K-fold and Leave-one-out, or a
        %   numeric scalar in Holdout and Resubstitution.
        TrainSize = [];
        
        %TESTSIZE Size of each test set.
        %   The TestSize property indicates the size of each test set. It
        %   is a numeric vector in K-fold and Leave-one-out, or a numeric
        %   scalar in Holdout and Resubstitution.
        TestSize = [];
    end
    
    properties(GetAccess = public, SetAccess = protected, Hidden=true)
         N = [];    
    end

    properties(GetAccess = public, SetAccess = protected, Dependent=true)
        %NUMOBSERVATIONS Number of observations.
        %   The NumObservations property is a numeric scalar holding the number of
        %   observations, including observations with missing GROUP values
        %   if GROUP is provided.
         NumObservations;
    end
    properties(GetAccess = public, SetAccess = public)
        indices = [];
        Group = [];
        holdoutT = [];
    end

    methods
        function n = get.NumObservations(this)
            n = this.N;
        end


        function  cv = repartition(cv,varargin)
        %REPARTITION Rerandomize a cross-validation partition. 
        %   D = REPARTITION(C) creates a new random cross-validation partition D
        %   of the same type and size as C.  Use REPARTITION to perform multiple
        %   Monte-Carlo repetitions of cross-validation.
        %   D = REPARTITION(C,S) uses the RandStream object S as its
        %   random number generator.
        %   
        %   See also CVPARTITION.

            if isempty(varargin)
                s = RandStream.getGlobalStream;
            else
                if length(varargin)>1
                    error(message('stats:cvpartition:RepartTooManyArg'));
                else
                    s = varargin{1};
                    if ~isa(s,'RandStream')
                        error(message('stats:cvpartition:RepartBadArg'));
                    end
                end
            end
        
            if strcmp(cv.Type,'resubstitution')
                warning(message('stats:cvpartition:RepartNone'));
                return;
            end
            %remove NaNs from cv.Group
            if ~isempty(cv.Group)
                [~,wasnan,cv.Group] = internal.stats.removenan(cv.Group);
                hadNaNs = any(wasnan);
            end

            %  regenerate the data partition
            cv = cv.rerandom(s);

            %add NaNs back into cv.indices and cv.Group
            if ~isempty(cv.Group) && hadNaNs
                [cv.indices, cv.Group] =...
                    internal.stats.insertnan(wasnan, cv.indices, cv.Group);
            end
        end % repartition
       
        function trainIndices = training(cv,i)
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
            switch cv.Type
                case {'kfold', 'leaveout'}
                    if nargin ~= 2
                        error(message('stats:cvpartition:WrongNumInputs'));
                    end
                    checkindex(i,cv.NumTestSets);
                    trainIndices = (cv.indices ~= i & ~isnan(cv.indices));

                case 'holdout'
                    if nargin == 2 && i~=1
                        error(message('stats:cvpartition:InvalidHOIndex'));
                    end
                    trainIndices = (cv.indices == 1);
                case 'resubstitution'
                    if nargin == 2 && i~= 1
                        error(message('stats:cvpartition:InvalidResubIndex'));
                    end
                    trainIndices = ~isnan(cv.indices);
            end
        end

        function testIndices = test(cv,i)
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
            switch cv.Type
                case {'kfold','leaveout'}
                    if nargin ~= 2
                        error(message('stats:cvpartition:WrongNumInputs'));
                    end
                    checkindex(i,cv.NumTestSets);

                    testIndices = (cv.indices == i);

                case 'holdout'
                    if nargin == 2 && i~= 1
                        error(message('stats:cvpartition:InvalidHOIndex'));
                    end
                    testIndices = (cv.indices == 2);
                case 'resubstitution'
                    if nargin == 2 && i ~= 1
                        error(message('stats:cvpartition:InvalidResubIndex'));
                    end
                    testIndices = ~isnan(cv.indices);
            end
        end

        % Display methods
%         function display(cv,objectname)
%             isLoose = strcmp(get(0,'FormatSpacing'),'loose');
% 
%             %objectname = inputname(1);
%             if isempty(objectname)
%                 objectname = 'ans';
%             end
% 
%             if (isLoose)
%                 fprintf('\n');
%             end
%             fprintf('%s = \n', objectname);
%             disp(cv);
%         end
        function disp(cv)
            isLoose = strcmp(get(0,'FormatSpacing'),'loose');

%             if (isLoose)
%                 fprintf('\n');
%             end
            switch cv.Type
                case 'kfold'
                    disp(getString(message('stats:cvpartition:KfoldCV')));
                case 'holdout'
                    disp(getString(message('stats:cvpartition:HoldoutCV')));
                case 'leaveout'
                    disp(getString(message('stats:cvpartition:LeaveoneoutCV')));
                case 'resubstitution'
                    disp (getString(message('stats:cvpartition:Resubstitution')));
            end
       
            fprintf('   NumObservations: %s\n',num2str(cv.NumObservations));
            fprintf('       NumTestSets: %s\n',num2str(cv.NumTestSets));
            Ndisp = 10;
            if cv.NumTestSets <= Ndisp
                 fprintf('         TrainSize: %s\n',num2str(cv.TrainSize ));
                 fprintf('          TestSize: %s\n',num2str(cv.TestSize ));

            else
                disp(['     TrainSize: ' num2str(cv.TrainSize(1:Ndisp)), ' ...']);
                disp(['      TestSize: ' num2str(cv.TestSize(1:Ndisp)), ' ...']);
            end
            %             end
        end
    end % public methods block


    methods(Access = 'protected')
        %re-generate the data partition using the RandStream object s
        function cv = rerandom(cv,s)

            switch cv.Type
                case 'kfold'
                    if isempty(cv.Group)
                        if cv.NumTestSets == cv.N
                            %special case of K-fold -- loocv
                            [~,cv.indices] = sort(rand(s,cv.NumTestSets,1)); % randperm 
                            cv.TestSize = ones(1,cv.NumTestSets);
                        else
                            cv.indices = kfoldcv(cv.N,cv.NumTestSets,s);
                            cv.TestSize = accumarray(cv.indices,1)';
                        end
                    else
                        if cv.NumTestSets == length(cv.Group)
                            %special case of K-fold -- loocv
                            [~,cv.indices] = sort(rand(s,cv.NumTestSets,1)); % randperm
                            cv.TestSize = ones(1,cv.NumTestSets);
                        else
                            cv.indices = stra_kfoldcv(cv.Group,cv.NumTestSets,s);
                            cv.TestSize = accumarray(cv.indices,1)';
                        end
                    end

                    cv.TrainSize = size(cv.indices,1) - cv.TestSize;

                case 'holdout'
                    if cv.holdoutT >= 1
                        if isempty(cv.Group)
                            cv.indices = holdoutcv(cv.N,cv.holdoutT,s);
                            cv.TestSize = cv.holdoutT;
                            cv.TrainSize = cv.N-cv.TestSize;
                        else
                            cv.indices = stra_holdoutcv(cv.Group, cv.holdoutT/length(cv.Group), s);
                            cv.TestSize = sum(cv.indices == 2);
                            cv.TrainSize = sum(cv.indices == 1);
                        end
                    else %hold cv.holdoutT*N out
                        if isempty(cv.Group)
                            cv.TestSize = floor(cv.N * cv.holdoutT);
                            cv.TrainSize = cv.N-cv.TestSize;
                            cv.indices = holdoutcv(cv.N,cv.TestSize,s);
                        else
                            cv.indices = stra_holdoutcv(cv.Group,cv.holdoutT,s);
                            cv.TestSize = sum(cv.indices == 2);
                            cv.TrainSize = sum(cv.indices == 1);
                        end
                    end

                case 'leaveout'
                    [~,cv.indices] = sort(rand(s,cv.NumTestSets,1));
            end
        end
    end % private methods block
    
    methods(Hidden = true)
        function b = fieldnames(a)
            b = properties(a);
        end
    end
   
end % classdef

%----------------------------------------------------
%stratified k-fold cross-validation
function cvid=stra_kfoldcv(group,nfold,s)
size_groups = accumarray(group(:),1);
if any(size_groups < nfold & size_groups > 0)
    warning(message('stats:cvpartition:KFoldMissingGrp'));
end
N = size(group,1);
cvid = 1 + mod((1:N)',nfold);
idrand = group + rand(s,N,1);
[~,idx] = sort(idrand);
cvid(idx) = cvid;
end

%----------------------------------------------------
%kfold cross-validation without stratification
function cvid = kfoldcv(N,nfold,s)
cvid = 1 + mod((1:N)',nfold);
[~,indices] = sort(rand(s,1,N)); % randperm
cvid = cvid(indices);
end

%-----------------------------------------------------
%holdout without stratification
function  idx= holdoutcv(N,num_test,s)
idx = 2*ones(N,1);
idx(1:N-num_test) = 1;
[~,indices] = sort(rand(s,1,N)); % randperm
idx = idx(indices);
end

%-----------------------------------------------------
%stratified holdout
function idx = stra_holdoutcv(group,test_ratio,s)
N = length(group);
size_groups = accumarray(group(:),1);
num_test = floor(size_groups * test_ratio);

test_diff = floor(N * test_ratio) - sum(num_test);
%add 1 for groups which are not in the test set
if any(num_test == 0)
    v=(num_test == 0);
    v(cumsum(v) > test_diff) = false;
    num_test(v) = num_test(v) + 1;
    test_diff = test_diff - sum(v);
end


if test_diff > 0
    ng= numel(size_groups);
    wasfull  =(num_test == size_groups);
    full_len = sum(wasfull);
    add = [ones(test_diff,1);zeros(ng - test_diff - full_len,1)];
    [~,indices] =  sort(rand(s,1,(ng-full_len))); % randperm
    add = add(indices);
    x = zeros(size(wasfull));
    x(~wasfull,:) = add;
    num_test = num_test + x;

end

if any(num_test == 0)
    warning(message('stats:cvpartition:HOTestZero'));
end

if any(num_test == size_groups)
    warning(message('stats:cvpartition:HOTrainingZero'));
end

idx = 2*ones(N,1);
for i = 1:numel(size_groups)
    g_idx = find(group == i);
    idx(g_idx(1:size_groups(i)-num_test(i))) = 1;
    [~,indices] = sort(rand(s,1,(size_groups(i)))); % randperm
    idx(g_idx) = idx(g_idx( indices ));
end

end

%-----------------------------
function checkindex(i,imax)
if ~(isnumeric(i) && isscalar(i) && i == round(i) && 1 <= i && i <= imax)
    error(message('stats:cvpartition:InvalidKFIndex'));
end
end

