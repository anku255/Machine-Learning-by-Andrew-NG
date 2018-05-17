## Table of Contents

* [Multiple Features](#multiple-features)
* [Gradient Descent for Multiple Variables](#gradient-descent-for-multiple-variables)
* [Gradient Descent in Practice I - Feature Scaling](#gradient-descent-in-practice-i---feature-scaling)
* [Gradient Descent in Practice II - Learning Rate](#gradient-descent-in-practice-ii---learning-rate)
* [Features and Polynomial Regression](#features-and-polynomial-regression)
* [Normal Equation](#normal-equation)
* [Normal Equation Noninvertibility](#normal-equation-noninvertibility)
* [How to plot J(theta_0, theta_1)](#normal-equation-noninvertibility)

## Multiple Features

![Multiple Features](/images/img15.png)

## Gradient Descent for Multiple Variables

![Gradient Descent for Multiple Variables](/images/img16.png)

## Gradient Descent in Practice I - Feature Scaling

![Gradient Descent in Practice I - Feature Scaling](/images/img17.png)

## Gradient Descent in Practice II - Learning Rate

![Gradient Descent in Practice II - Learning Rate](/images/img18.png)

## Features and Polynomial Regression

![Features and Polynomial Regression](/images/img19.png)

## Normal Equation

![Normal Equation](/images/img20.png)

## Normal Equation Noninvertibility

![Normal Equation Noninvertibility](/images/img21.png)

## How to plot J(theta_0, theta_1)

The `computeCost` function takes dataset points and the theta values and returns the cost for the given theta values.

```
function J = computeCost(X, y, theta)
  m = length(y); % number of training examples
  J = 0;
  error = (X*theta - y);
  sqError = error.^2;
  summation = sum(sqError);
  J = summation/(2*m);
end
```

Now, we want to plot the cost value for a lot of values of theta_0 and theta_1. Therefore, the first step is to generate the values of theta_0 and theta_1.

```
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
```

`linspace(start, end, N)` function generates a row vector of `N` equally spaced values in the between `start` and `end`. In this case, we will have 100 equally spaced values for theta_0 and theta_1 in their respective range.

Now, for every combination of theta_0 and theta_1 we want to `computeCost` and store it in a matrix. So, let's create a 2D matrix which will store the cost values.

```
J_vals = zeros(length(theta0_vals), length(theta1_vals));
```

Now, we let's compute and store the cost in `J_vals` for every combination of theta_0 and theta_1.

```
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end
```

The two for loops are doing nothing but just calculating every possible combination of theta_0 and theta_1 which is stored in `t`. We call `computeCost` for the current combination and store it in `J_vals`.

Now, the only thing left to do is to plot `J_vals`. We will need to transpose `J_vals` because of the way `meshgrids` work in the `surf` command. If we don't do that, then the axes will be flipped.

```
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');
```

The plot would something like below:
![Surface Plot](/images/img22.png)

We can also plot the contour plot.

```
% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
```

![Contour Plot](/images/img23.png)
