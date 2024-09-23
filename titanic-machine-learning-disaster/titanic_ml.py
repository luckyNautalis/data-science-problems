r"""°°°
#### Goal
- Predict if a passenger survived the sinking of the Titanic or not (classification task), this prediction is either a value of 0 or 1.
#### Metric
Accuracy: The ratio of correct predictions out of all predictions.
#### Data Summary
- train.csv contains details of passengers on board, and whether they survived (use this to predict survivorship of passengers in test.csv).
- test.csv contains details of passengers on board without info on whether they survived.
#### Submit
- File w/ 2 columns:
  - PassengerId (in any sorted order)
  - Survived (binary predictions 0/1)
°°°"""
# |%%--%%| <xzCy6Gykch|Sptd7LFsQs>

import copy
import math

import numpy as np
import pandas as pd

titanic_data = pd.read_csv('train.csv')

# |%%--%%| <Sptd7LFsQs|wLy0RRacK5>

titanic_data.head()

# |%%--%%| <wLy0RRacK5|P0U7z2nLrh>

titanic_data.info()

# |%%--%%| <P0U7z2nLrh|CndxX8QmFa>

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# |%%--%%| <CndxX8QmFa|qe7H7Y9fH7>

X = titanic_data[features]
y = titanic_data['Survived']

# |%%--%%| <qe7H7Y9fH7|AfWTTzl8rT>

# Cleaning Data
X = X.dropna()
X.loc[:, 'Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X.loc[:, 'Embarked'] = X['Embarked'].map({'C': 0, 'S': 1, 'Q': 2})
X.head()

# |%%--%%| <AfWTTzl8rT|Buv1LVXIEK>

X.dtypes

# |%%--%%| <Buv1LVXIEK|3LzmnG6m70>

# Converting to number types
X['Sex'] = X['Sex'].astype(int)
X['Embarked'] = X['Embarked'].astype(int)

# |%%--%%| <3LzmnG6m70|qiKgzi4vuu>

X_train = X.values
y = y.values

# |%%--%%| <qiKgzi4vuu|V2ks28c3Ef>

X.shape

# |%%--%%| <V2ks28c3Ef|Cqe4WaAXWb>
r"""°°°
#### Hypothesis

$$\hat{y} =\sigma(W^TX + b)$$
°°°"""
# |%%--%%| <Cqe4WaAXWb|pVhqsl08k3>
r"""°°°
Repeat $(*)$ until convergence 
$$
\begin{align*}
&\begin{cases}
w_{j} := w_{j} - \alpha\frac{\partial J(\vec{w},b)}{\partial w_{j}}, \quad 0\leq j\leq n-1 \\
\ \ \vdots \\
b := b - \alpha\frac{\partial J(\vec{w},b)}{\partial b} \\
\end{cases}\tag{*} \\ \\
&\frac{\partial J(\vec{w},b)}{\partial w_{j}} = \frac{1}{m}\sum^{m-1}_{i=0}\left(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}\right)x_{j}^{(i)}\tag{Gradient w.r.t. $w_{j}$}\\
&\frac{\partial J(\vec{w},b)}{\partial b}=\frac{1}{m}\sum_{i=0}^{m-1}\left(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}\right)\tag{Gradient w.r.t. $b$}\\
&J(\vec{w},b) = -\frac{1}{m}\sum_{i=0}^{m-1}L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) \tag{Logistic Cost Function}  \\
&L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) = \left[y^{(i)}\log\left({f_{\vec{w},b}(\vec{x}^{(i)})}\right) + (1-y^{(i)})\log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right)\right]\tag{Logistic Loss Function}
\end{align*}
$$

- $f_{\vec{w},b}(\vec{x}^{(i)})$  is the logistic regression  function.
°°°"""
# |%%--%%| <pVhqsl08k3|FFp1WKvXcK>

def sigmoid(z):
    """
    Compute the sigmoid of z (loss of z)

    Args:
        z (ndarray): A scalar, numpy array of any size

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    return 1/(1+np.exp(-z))

# |%%--%%| <FFp1WKvXcK|jZcb9Z2gyZ>

def logistic_cost(X, y, w, b):
    """
    Computes cost with cross-entropy loss function

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        sig_loss_i = sigmoid(z_i)
        cost += -y[i]*np.log(sig_loss_i) - (1-y[i])*np.log(1-sig_loss_i)
    # Divide by m to get average loss
    return cost/m

# |%%--%%| <jZcb9Z2gyZ|FImBw6gMRV>

def logistic_gradient(X, y, w, b): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        # Make a prediction for example i
        f_wb_i = sigmoid(np.dot(X[i], w) + b) # (n,)(n,)=scalar
        # Compute error
        err_i = f_wb_i - y[i]
        # Compute gradient w.r.t. weights
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (err_i * X[i, j])
        # Compute gradient w.r.t. bias
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

# |%%--%%| <FImBw6gMRV|p22KrHpeGL>

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # Array to store cost J and w's at each iteration
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        # Calculate gradient and update parameters
        dj_dw, dj_db = logistic_gradient(X, y, w, b)

        # Update weights and bias
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        # Save cost J at each iteration
        if i < 10_000: # prevents exhausting resources
            J_history.append(logistic_cost(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b

# |%%--%%| <p22KrHpeGL|2AN5F0tMb8>

w, b = gradient_descent(X_train, y, np.zeros_like(X_train[0]), 0.0, 0.001, 100)

# |%%--%%| <2AN5F0tMb8|EHUDAGvfaQ>

z = np.dot(X_train, w) + b
y_pred = sigmoid(z)
class_pred = np.where(y_pred<0.5, 0, 1)
class_pred
