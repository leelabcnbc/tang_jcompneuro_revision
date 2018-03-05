function dev = dev_poisson(mu,y)
%DEV_POISSON Summary of this function goes here
%   Detailed explanation goes here
% mu is inferred value.
% y is truth.
dev = 2*(y .* (log((y+(y==0)) ./ mu)) - (y - mu));
end

