classdef cvpartitionInMemoryImpl < cvpartitionImpl
   
methods
        function cv = cvpartitionInMemoryImpl(N,method,T,varargin)
            
            if nargin>0 && isstruct(N) && isfield(N,'BckCmpBackdoorConstructor')
                % only loadobj uses this backdoor constructor for backward
                % compatibility
                N = rmfield(N,'BckCmpBackdoorConstructor');
                fn = fieldnames(N);
                for ii=1:numel(fn)
                    cv.(fn{ii}) = N.(fn{ii});
                end
                return
            end
                
            if isempty(varargin)
                s = RandStream.getGlobalStream;
                stdargin = nargin;
            else
                if length(varargin)>1
                    error(message('stats:cvpartition:cvpartTooManyArg'));
                else
                    stdargin = nargin-1;
                    s = varargin{1};
                    if ~isa(s,'RandStream')
                        error(message('stats:cvpartition:cvpartBadArg'));
                    end
                end
            end
            
            if stdargin < 2
                error(message('stats:cvpartition:TooFewInputs'));
            end

            if ischar(method) && size(method,1) == 1
                methodNames = {'kfold','holdout','leaveout','resubstitution'};
                j = find(strncmpi(method,methodNames,length(method)));
                if length(j) > 1
                    error(message('stats:cvpartition:AmbiguousMethod', method));
                elseif isempty(j)
                    error(message('stats:cvpartition:UnknownMethod'));
                end
            else
                error(message('stats:cvpartition:InvalidType'));
            end

            cv.Type = methodNames{j};

            if isscalar(N)
                if ~isnumeric(N) || N <= 1 || N ~= round(N) || ~isfinite(N)
                    error(message('stats:cvpartition:BadNX'));      
                end
                cv.N = N;
            else
                cv.Group = grp2idx(N);
                cv.N = length(cv.Group); % the number of observations including NaNs
                [~,wasnan,cv.Group] = internal.stats.removenan(cv.Group);
                hadNaNs = any(wasnan);
                if hadNaNs
                    warning(message('stats:cvpartition:MissingGroupsRemoved'));
                    if length (cv.Group) <= 1
                        error(message('stats:cvpartition:BadNGrp'));
                    end
                end
            end

            dftK = 10; % the default number of subsamples(folds) for Kfold
            P  = 1/10; % the default holdout ratio

            switch cv.Type
                case 'kfold'
                    if stdargin == 2 || isempty(T)
                        T = dftK;
                    elseif ~isscalar(T) || ~isnumeric(T) || T <= 1 || ...
                            T ~= round(T) || ~isfinite(T)
                        error(message('stats:cvpartition:BadK'));
                    end

                    if  isempty(cv.Group) && T > cv.N
                        warning(message('stats:cvpartition:KfoldGTN'));
                        T = cv.N;
                    elseif ~isempty(cv.Group) && T > length(cv.Group)
                        warning(message('stats:cvpartition:KfoldGTGN'));
                        T = length(cv.Group);
                    end

                    cv.NumTestSets = T; %set the number of fold
                    cv = cv.rerandom(s);

                case 'leaveout'
                    if stdargin > 2 && T ~= 1
                        error(message('stats:cvpartition:UnsupportedLeaveout'));
                    end

                    if isempty(cv.Group)
                        cv.NumTestSets = cv.N;
                    else
                        cv.NumTestSets = length(cv.Group);
                    end

                    [~,cv.indices] = sort(rand(s,cv.NumTestSets,1));

                    cv.TrainSize = (cv.NumTestSets-1) * ones(1,cv.NumTestSets);
                    cv.TestSize = ones(1,cv.NumTestSets);

                case 'resubstitution'
                    if stdargin > 2 
                        error(message('stats:cvpartition:UnsupportedCV'));
                    end

                    if isempty(cv.Group)
                        numObs = N;
                    else
                        numObs = length(cv.Group);
                    end

                    cv.indices = (1: numObs)';
                    cv.NumTestSets = 1;
                    cv.TrainSize =  numObs;
                    cv.TestSize =  numObs;

                case 'holdout'
                    if stdargin == 2 || isempty(T)
                        T = P;
                    elseif ~isscalar(T) || ~ isnumeric(T) || T <= 0 || ~isfinite(T)
                        error(message('stats:cvpartition:BadP'));
                    end

                    if T >= 1 %hold-T observations out
                        if T ~=round(T)
                            error(message('stats:cvpartition:BadP'));
                        end
                        if isempty(cv.Group)
                            if T >= cv.N
                                error(message('stats:cvpartition:PNotLTN'));
                            end
                        else
                            if T>= length(cv.Group)
                                error(message('stats:cvpartition:PNotLTGN'));
                            end
                        end
                    else
                        if (isempty(cv.Group) && floor(cv.N *T) == 0) ||...
                                (~isempty(cv.Group) && floor(length(cv.Group) * T) == 0)
                            error(message('stats:cvpartition:PTooSmall'));

                        end
                    end

                    cv.holdoutT = T;
                    cv = cv.rerandom(s);
                    cv.NumTestSets = 1;
            end

            %add NaNs back
            if ~isempty(cv.Group) && hadNaNs
                [cv.indices, cv.Group] =...
                    internal.stats.insertnan(wasnan, cv.indices, cv.Group);
            end
        end % cvpartition constructor    
    
end
    
end