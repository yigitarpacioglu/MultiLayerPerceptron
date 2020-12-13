function p = pred(w1, w2, X)
% this function predict the label of an input given a trained neural network

z2 = X * w1';
a2 = [ones(size(sigmoid(z2), 1), 1) sigmoid(z2)];

% H_theta(x)
z3 = a2 * w2';
a3 = sigmoid(z3);

[x, ix] = max(a3, [], 2);

p = ix;

% =========================================================================


end