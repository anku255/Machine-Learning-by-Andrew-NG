## Linear Regression

### Hypothesis function

Hypothesis function for a set of training examples having multiple feature in each example (n >= 1) is given as below:

![Hypothesis function](/images/img24.png)

#### Vectorized Equation to compute Hypothesis

`h = theta*X`

Where `theta` is a **m x 1 column vector** and `X` is a **m x n matrix**.

### Cost Function

Cost function is a function of `theta`. For univeriate linear regression the cost function is given as below:
![Cost function](/images/img25.png)

For multiveriate linear regression, we will have more theta values.

#### Vectorized Equation to compute Cost

`J = ( (X*theta - y)'*(X*theta - y) ) / (2*m);`

Cost could also be computed in a non vectorized form as below but the above formula gives the same result in more concise manner.

```matlab
error = (X*theta - y);
sqError = error.^2;
summation = sum(sqError);
J = summation/(2*m);
```

### Gradient Descent

The general formula to calculate gradient descent is given below:
![Gradient Descent](/images/img26.png)

After simplifying the partial derivative, we will get a formula as given below:
![Gradient Descent Simplified](/images/img27.png)

#### Vectorized Equation to compute Gradient Descent

`theta = theta - (alpha/m)*(X')*(X*theta - y);`

### Feature Normalization

To make gradient descent run faster, we can normalize each feature of our training example. The general formula for feature normalization is given below:

![Feature Normalization](/images/img28.png)

**Note:** Feature Normalization is applied column vise.

#### Vectorized Equation for Feature Normalization

```matlab
mu = mean(X);
sigma = std(X);
X_norm = (X - mu)./sigma;
```

**Note:** Here, X is a multidimension matrix and we are apply feature normalization to every column in one go.

### Normal Equation

The formula to caluculate normal equation is given below:
</br>
![Normal Equation](/images/img29.png)

#### Vectorized Equation for Normal Equation

`theta = pinv( (X'*X) ) * X' * y`