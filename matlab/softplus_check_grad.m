function softplus_check_grad()
%SOFTPLUS_CHECK_GRAD Summary of this function goes here
%   Detailed explanation goes here

input_range = linspace(0.1, 2, 20000);
output_range = funcs.inv_softplus(input_range);
d_numerical = diff(output_range)./diff(input_range);
d_symbolic = funcs.inv_softplus_d(input_range(1:end-1));
d_symbolic_2 = funcs.inv_softplus_d(input_range(2:end));
disp(corr(d_numerical(:), d_symbolic(:)));
disp(d_numerical(1:10));
disp(d_symbolic(1:10));
close all;
figure;
hold on;
plot(d_numerical(1:10));
plot(d_symbolic(1:10));
plot(d_symbolic_2(1:10));
legend('numerical', 'symb 1', 'symb 2');
hold off;
end

