function H = Hankel(x, n, L, T)
% definition of Hankel matrix

H = zeros(T * n, L - T + 1);
for i = 1 : T
    H((i - 1) * n + 1 : i * n, :) = x(:, i : i + L - T);
end
