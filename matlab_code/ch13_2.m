n = 100;
x = [2 * randn(n, 1) randn(n, 1)];
%x = [2 * randn(n, 1) 2 * round(randn(n, 1)) - 1 + randn(n, 1) / 3];
x = x - repmat(mean(x), n, 1);
[t, v] = eigs(x' * x, 1);
figure(1); clf; hold on; axis([-6 6 -6 6])
plot(x(:, 1), x(:, 2), 'rx');
plot(9 * [-t(1) t(1)], 9 * [-t(2) t(2)]);