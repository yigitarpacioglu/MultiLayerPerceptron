function g = sig_grad(z)
%returns the gradient of the sigmoid function

g = zeros(size(z));

g = sigmoid(z) .* (1 - sigmoid(z));


end
