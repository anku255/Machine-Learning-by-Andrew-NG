function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% convert labels in y vector to matrix which contain only 0 and 1 values
% for example a row having value 5 will be converted to [0 0 0 0 1 0 0 0 0 0]
y_matrix = eye(num_labels)(y,:);

% calculate forward propagation

% add a column of ones in X (bias unit)
a1 = [ones(m,1), X]; % 5000x401
z2 = a1*Theta1'; % (5000x401)*(401*25) => (5000x25)
a2 =  sigmoid(z2); % (5000x25)
a2 = [ones(m,1), a2]; % (5000x26)
z3 = a2*Theta2'; % (5000x26)*(26x10) => (5000x10)
a3 = sigmoid(z3); % (5000x10)

% J = (1x1) * sum(sum( (5000x10).* (5000x10) - (5000x10).*(5000x10)))
% J = (1x1) * sum(sum( 5000x10))
% J = (1x1) * (1x1) => 1x1
J = (1/m) * sum(sum( -y_matrix.*log(a3) - (1-y_matrix).*log(1-a3) ));

% remove first column from both Theta1 and Theta2 as we do not 
% regularize Theta values used for bias units
regTheta1 = Theta1(:,2:end); % (25x400)
regTheta2 = Theta2(:,2:end); % (10x25)

% regularizedTerm = (1x1) * (1x1) => 1x1
regularizedTerm = (lambda/(2*m)) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));

% regularize the cost
J = J + regularizedTerm; % (1x1)

% Backpropagation

del1 = zeros(size(Theta1)); % (25x401)
del2 = zeros(size(Theta2)); % (10x26)

for t = 1:m
   a1t = X(t,:); % (1x400)
   a1t = [1, a1t]; % (1x401)
   
   z2t = a1t*Theta1'; % (1x401)*(401x25) => (1x25)
   a2t = sigmoid(z2t); % (1x25)
   a2t = [1, a2t]; % (1x26)
   
   z3t = a2t*Theta2'; % (1x26)*(26x10) => (1x10)
   a3t = sigmoid(z3t); % (1x10)
    
   yt = y_matrix(t, :); % (1x10)
  
   d3 = a3t - yt; % (1x10)-(1x10) => (1x10)
   
   % d2 = (26x10)x(10x1) .* (1x26)'
   % d2 = (26x1) .* (26x1)
   % d2 = (26x1)
   d2 = (Theta2' * d3') .* sigmoidGradient([1 z2t])';
   
   d2 = d2(2:end); % (25x1)
  
   % del1 = (25x401) + (25x1)* (1x401)
   % del1 = (25x401)
   del1 = del1 + d2*a1t;
   
   % del2 = (10x26) + (1x10)'*(1x26)
   % del2 = (10x26) + (10x1)*(1x26) => (10x26)
   del2 = del2 + d3'*a2t;
   
endfor

Theta1_grad = (1/m) * del1;
Theta2_grad = (1/m) * del2;

% Regularize Gradient
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
