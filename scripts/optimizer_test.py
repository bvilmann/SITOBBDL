import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

# Define the function to be optimized and its gradient
def f(x, a, b):
    return a*(x[0] - 1)**2 + b*(x[1] - 2.5)**2

def grad_f(x, a, b):
    return np.array([2*a*(x[0] - 1), 2*b*(x[1] - 2.5)])

# Define the Hessian matrix as a callable function with arguments
def hess_f(x, a, b):
    return np.array([[2*a, 0], [0, 2*b]])

# Define the initial guess and additional arguments
x0 = np.array([0, 0])
a = 2
b = 2

# Call the minimize function with the pre-calculated Hessian matrix and arguments
result = minimize(f, x0, args=(a, b), method='Newton-CG', jac=grad_f, hess=hess_f, options={'xtol':1e-8,'disp': True})

# Print the optimization result
print(result)
t = np.linspace(0,3,100)
fx = np.zeros((100,100))
for i,v1 in enumerate(t):
    for j,v2 in enumerate(t):
        fx[i,j] = f(np.array([v1, v2]),a,b)


# Verify solution index:
df = pd.DataFrame(fx, columns=t, index=t)
df.min(axis=1)

# plot
fig, ax = plt.subplots(1,1,dpi=200)
ax.imshow(fx.T,extent=(0,3,3,0))
ax.scatter(result.x[0],result.x[1],color='red',marker='x')

# fig, ax = plt.subplots(2,1,dpi=200)
# data = fx[75,:]
# ax[0].plot(t,data/max(data),label='log')
# ax[0].plot(t,np.exp(data)/max(np.exp(data)),label='normal')
# ax[0].legend()
# ax[1].plot(t[1:],abs(np.diff(data/max(data))))
# ax[1].plot(t[1:],abs(np.diff(np.exp(data)/max(np.exp(data)))))

