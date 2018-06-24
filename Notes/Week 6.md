- [Evaluation a Learning Algorithm](#evaluation-a-learning-algorithm)
  - [Evaluating a Hypothesis](#evaluating-a-hypothesis)
  - [Model Selection and Train/Validation/Test Sets](#model-selection-and-trainvalidationtest-sets)
  - [Diagnosing Bias vs Variance](#diagnosing-bias-vs-variance)
  - [Regularization and Bias/Variance](#regularization-and-biasvariance)
  - [Learning Curves](#learning-curves)
  - [Deciding What to Do Next Revisited](#deciding-what-to-do-next-revisited)
  - [Prioritizing What to Work On](#prioritizing-what-to-work-on)
  - [Error Analysis](#error-analysis)
  - [Error Metrics for Skewed Classes](#error-metrics-for-skewed-classes)
  - [Trading Off Precision and Recall](#trading-off-precision-and-recall)

## Evaluation a Learning Algorithm

### Evaluating a Hypothesis

![Evaluating a Hypothesis](/images/img60.png)

### Model Selection and Train/Validation/Test Sets

![Model Selection and Train/Validation/Test Sets](/images/img61.png)

---

### Diagnosing Bias vs Variance

![Diagnosing Bias vs Variance](/images/img62.png)

### Regularization and Bias/Variance

![Regularization and Bias/Variance](/images/img63.png)

### Learning Curves

![Learning Curves](/images/img64.png)

### Deciding What to Do Next Revisited

![Deciding What to Do Next Revisited](/images/img65.png)

---

### Prioritizing What to Work On

![Prioritizing What to Work On](/images/img66.png)

### Error Analysis

![Error Analysis](/images/img67.png)

### Error Metrics for Skewed Classes

**Skewed Training Data:** In a binary classification case, skewed data means that one class is vastly more represented in the data than the other class.

In the example below, we can see that even a bad/naive algorithm can have a very high accuracy in case of a skewed data set.

![Skewed data example](/images/img68.png)

Instead of using Test Set Error as a peformance metric, we use **Precision** and **Recall** to benchmark an algorithm in case of a skewed training data.

![Precision and Recall](/images/img69.png)

### Trading Off Precision and Recall

As we shall see below, choosing a **high threshold value** will lead to **higher precison** and **lower recall** whereas a **low threshold value** means **lower precision** but **higher recall**.

![Precision and Recall tradeoff](/images/img70.png)

Clearly, having a higher precison or higher recall alone doesn't tell us how good an algorithm is. The algorithm should be considered good when its **F1 Score** is high.

F1 Score combines both Precision and Recall and ensures that if either one of precision or recall is low then its score will also be low.

![F1 Score](/images/img71.png)
