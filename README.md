# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\50_Startups.csv")

# Let's assume 'R&D Spend' predicts 'Profit'
X = df['R&D Spend'].values
y = df['Profit'].values

# Normalize data (important for gradient descent)
X = (X - X.mean()) / X.std()

# Add bias term (x0 = 1)
m = len(X)
X = np.c_[np.ones(m), X]

# Initialize parameters
theta = np.zeros(2)

# Hyperparameters
alpha = 0.01   # learning rate
iterations = 1000

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    return (1/(2*m)) * np.sum((predictions - y) ** 2)

# Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y

        gradients = (1/m) * X.T.dot(errors)
        theta = theta - alpha * gradients

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Train model
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

print("Final Parameters (theta):", theta)

# Plot cost function
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations")
plt.show()

# Predictions
predictions = X.dot(theta)

# Plot regression line
plt.scatter(X[:,1], y, color='blue')
plt.plot(X[:,1], predictions, color='red')
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.show()
```

## Output:
<img width="1020" height="733" alt="Screenshot 2026-05-11 154428" src="https://github.com/user-attachments/assets/7eec5636-a06d-4216-98b7-154455f58d02" />

<img width="986" height="684" alt="Screenshot 2026-05-11 154505" src="https://github.com/user-attachments/assets/b395c448-c69e-4ad9-81ae-7827cc307d23" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
