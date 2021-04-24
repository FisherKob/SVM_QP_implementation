#!/usr/bin/env python
# coding: utf-8

# In this module, I will implement Support Vector Machine (SVM) using the conect of convex optimization.
# 
# Suppose we have data sample $\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{n}$ with $y_{i} \in\{-1,+1\}$. We know the support vector machine is an example of the primal/dual optimization problem.
# 
# \begin{array}{c}
# \min _{\left(\beta_{0}, \boldsymbol{\beta}, \boldsymbol{\xi}\right)} \frac{1}{2}\|\boldsymbol{\beta}\|_{2}^{2}+\lambda \sum_{i=1}^{n} \xi_{i} \\
# \text { subject to } \quad \xi_{i} \geq 0, \quad i=1,2, \cdots, n \\
# y_{i}\left(\mathbf{x}_{i}^{T} \boldsymbol{\beta}+\beta_{0}\right) \geq 1-\xi_{i}
# \end{array}
# 
# 
# Define the dual variables $\nu=\left(\nu_{1}, \cdots, \nu_{n}\right)^{T} \geq 0$ and $\mathbf{u}=\left(u_{1}, \cdots, u_{n}\right)^{T} \succeq 0$. Also let $\mathbf{y}=\left(y_{1}, y_{2}, \cdots, y_{n}\right)^{T}$ and
# $$
# \tilde{X}=\left(\begin{array}{c}
# y_{1} \mathbf{x}_{1}^{T} \\
# y_{2} \mathbf{x}_{2}^{T} \\
# \vdots \\
# y_{n} \mathbf{x}_{n}^{T}
# \end{array}\right)
# $$
# 
# Then through deriving Lagrangian function and dual function, we obtain the dual problem
# $$
# \begin{array}{c}
# \max _{\mathbf{u}}-\frac{1}{2} \mathbf{u}^{T} \tilde{X} \tilde{X}^{T} \mathbf{u}+\mathbf{1}_{n}^{T} \mathbf{u} \\
# \text { subject to } \quad \mathbf{u} \geq \mathbf{0}, \\
# \mathbf{u} \leq \lambda \mathbf{1}_{n} \\
# \mathbf{y}^{T} \mathbf{u}=0
# \end{array}
# $$
# which is equivalent to
# $$
# \max _{\mathbf{u}}-\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} u_{i} u_{j} y_{i} y_{j}\left\langle\mathbf{x}_{i}, \mathbf{x}_{j}\right\rangle+\sum_{i=1}^{n} u_{i}
# $$
# $$
# \begin{array}{l}
# \text { subject to } \quad 0 \leq u_{i} \leq \lambda, \quad i=1,2, \cdots, n \\
# \sum_{i=1}^{n} y_{i} u_{i}=0,
# \end{array}
# $$
# where $\left\langle\mathbf{x}_{i}, \mathbf{x}_{j}\right\rangle=\mathbf{x}_{i}^{T} \mathbf{x}_{j}$ is the inner product.
# Clearly, the dual problem above is a quadratic programmming (QP) problem, and we have learned how to solve QP. Now denote $\boldsymbol{\nu}^{*}=\left(\nu_{1}^{*}, \cdots, \nu_{n}^{*}\right)^{T}$ and $\mathbf{u}^{*}=\left(u_{1}^{*}, \cdots, u_{n}^{*}\right)^{T}$ be the dual optimal point. Then by KKT condition, we can obtain the primal optimal point for $\boldsymbol{\beta}$ that
# $$
# \boldsymbol{\beta}^{*}=\tilde{X}^{T} \mathbf{u}^{*}=\sum_{i=1}^{n} u_{i}^{*} y_{i} \mathbf{x}_{i}
# $$
# Also, if $0<u_{i}^{*}<\lambda$, then by the complemenatry slackness condition in KKT, we have that $\nu_{i}^{*}>0, \xi_{i}^{*}=0$, and
# $$
# \beta_{0}^{*}=y_{i}-\mathbf{x}_{i}^{T} \boldsymbol{\beta}^{*}
# $$
# Now let's implement the optimization derivation above into Python code. First we import some relevant packages and generate some data sample to play with. We refer tot he following for data generation process:
# $$
# P\left(y_{i}=+1 \mid \mathbf{x}_{i}\right)=1-P\left(y_{i}=-1 \mid \mathbf{x}_{i}\right)=\frac{\exp \left(\mathbf{x}_{i}^{T} \boldsymbol{\beta}\right)}{1+\exp \left(\mathbf{x}_{i}^{T} \boldsymbol{\beta}\right)}, \\
# with \ \boldsymbol{\beta}=\sigma(1,...,1)^T \in ℝ
# $$
# 

# Next we define a core function to implement SVM using 𝚌𝚟𝚡𝚘𝚙𝚝 package

# In[173]:


import numpy as np
import math
from cvxopt import matrix
from cvxopt import solvers
import time

# generate data sample for illustration
np.random.seed(999) # set an arbitrary random seed
n1 = 400 
p = 20
mu = np.zeros(p)
Sigma = np.diag(np.ones(p))
X = np.random.multivariate_normal(mu, Sigma, n1)
signal = 2.0
beta = np.ones(p)*signal
prob = 1.0/(1+np.exp(-X.dot(beta)))
y = (prob < np.random.uniform(0, 1, n1))*2.0-1


# split the data (X,y) to traning set and testing set
n = math.floor(n1/2)
X_train = X[:n]
X_test = X[n:]
y_train = y[:n]
y_test = y[n:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



# define a function to train SVM
def svmfit(X, y, lam):
    n, p = X.shape
    # Gram matrix from (linear) kernel function
    Gmat = X.dot(np.transpose(X))

    # define matrices/vectors for solving dual problem with QP
    P = matrix(np.outer(y, y)*Gmat)

    q = matrix(-1.0*np.ones(n))
    A = matrix(y, (1, n))
    b = matrix(0.0)

    temp1 = -np.identity(n)
    temp2 = np.identity(n)
    G = matrix(np.vstack((temp1, temp2)))
    temp1 = np.zeros(n)
    temp2 = np.ones(n)*lam
    h = matrix(np.concatenate((temp1, temp2)))

    # solve QP
    sol = solvers.qp(P, q, G, h, A, b)
    # Find dual optimal point for Lagrange multiplier
    u = np.ravel(sol['x'])
    print("there are %d support points out of %d points" % (np.sum(np.logical_and(u>1e-5,(lam-u)>1e-5)), n))

    # compute beta
    tildeX = np.reshape(np.repeat(y,p),(n,p),order='C')*X
    tildeXt = tildeX.transpose()
    beta = tildeXt.dot(u)

    # Find data points that lie in margin's boundary
    inds = np.logical_and((u>1e-5),((lam-u)>1e-5))
    # compute beta0
    beta0 = np.mean((y[inds]-X[inds,:].dot(beta)))

    return beta0, beta, u


# In[174]:


lam = 1.0

start = time.time()
res = svmfit(X_train,y_train,lam)

timing1 = time.time()-start
beta0 = res[0]
beta = res[1]
print("Computing time for our implemented SVM is %s s" % timing1)


# In[175]:


y_train_fitted = np.sign(X_train.dot(beta)+beta0)
err_fitted = np.sum(np.abs(y_train-y_train_fitted)>1)/len(y_test)

# use testing data to check prediction performance
y_test_pred = np.sign(X_test.dot(beta)+beta0)
err_testing = np.sum(np.abs(y_test-y_test_pred)>1)/len(y_test)

print("Model fitting error rate for training data is %5.2f%%.\n" % (err_fitted*100))


# In[176]:


print("Model prediction error rate for testing data is %5.2f%%.\n" % (err_testing*100))


# In[177]:


#Now we use sklearn svm classifer to implement the same
from sklearn import svm
start = time.time()
clf = svm.SVC(C=1.0,kernel='linear')
clf.fit(X_train,y_train)


# In[178]:


timing2 = time.time()-start
print("Computing time using Scikit-learn package is %f s" % timing2)


# In[179]:


print("QP implementation gives estimate:")


# In[180]:


print(beta)


# In[181]:


print(beta0)


# In[172]:


# Our sklearn SVM implementation
print(clf.coef_)

