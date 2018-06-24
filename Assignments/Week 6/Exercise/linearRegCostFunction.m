function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% (X*theta - y) => (mx2)*(2x1) - (mx1) => (mx1)
% (X*theta - y)' => (1xm)
% J =  (1x1) * (1xm)*(mx1) => (1x1)
J = 1/(2*m)*((X*theta - y)' * (X*theta - y));

% RegularizationTerm = (1x1)
RegularizationTerm =  (lambda/(2*m))*sum(theta(2:end).^2);

% J = (1x1) + (1x1) => (1x1)
J = J + RegularizationTerm;


% grad = (1x1) * (mx2)' * (mx1)
% grad = (1x1) * (2x1) => (2x1) scalar multiplication
grad = (1/m)*X'*(X*theta -y);

grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);
% =========================================================================

grad = grad(:);

end
