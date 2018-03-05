function s = softplus_handles()
%SOFTPLUS_HANDLES Summary of this function goes here
%   Detailed explanation goes here
s = struct();
s.Link = @funcs.inv_softplus;
s.Derivative = @funcs.inv_softplus_d;
s.Inverse = @funcs.softplus;
end

