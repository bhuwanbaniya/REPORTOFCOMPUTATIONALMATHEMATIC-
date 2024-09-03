import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#BHUWANBANIYA
#2414002
def plot_function(func, a, b):
    """
    This function plots the graph of the input func
    within the given interval [a,b).
    """
    x = np.linspace(a, b, 1000)
    y = func(x)
    plt.plot(x, y, label='Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Graph of the function')
    plt.grid(True)

def newton_method(func, grad, x0, tol=1e-6, max_iter=100):
    """
    Approximate solution of f(x)=0 by Newton-Raphson's method.

    Parameters:
    - func: Function for which we are searching for a solution f(x)=0.
    - grad: Gradient of the function f(x).
    - x0: Initial guess for a solution f(x)=0.
    - tol: Tolerance level for checking convergence of the method.
    - max_iter: Maximum number of iterations of Newton's method.

    Returns:
    - root: Approximation of the root.
    - iterations: Number of iterations taken to find the root.
    """
    iter_count = 1
    while iter_count <= max_iter:
        if np.abs(grad(x0)) <= 1e-12:
            print("Mathematical Error! Found root may not be correct.")
            return x0, iter_count
        x1 = x0 - func(x0) / grad(x0)
        if np.abs(func(x1)) <= tol:
            return x1, iter_count
        x0 = x1
        iter_count += 1

    print("Warning! Method exceeded maximum number of iterations.")
    return None, max_iter

if _name_ == "_main_":
    func1 = lambda x: x**2 - x - 1
    grad1 = lambda x: 2*x - 1

    func2 = lambda x: x*3 - x*2 - 2*x + 1
    grad2 = lambda x: 3*x**2 - 2*x - 2

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plot_function(func1, -1, 2)  # Using the same interval for both functions
    plt.legend(['Function 1'])

    plt.subplot(1, 2, 2)
    plot_function(func2, -1.5, 2)  # Using the same interval for both functions
    plt.legend(['Function 2'])

    plt.tight_layout()
    plt.show()

    root1_newton, iterations_1 = newton_method(func1, grad1, 1)
    root2_newton, iterations_2 = newton_method(func1, grad1, -1)

    root1_newton_2, iterations_3 = newton_method(func2, grad2, 0)
    root2_newton_2, iterations_4 = newton_method(func2, grad2, -1)
    root3_newton_2, iterations_5 = newton_method(func2, grad2, 2)

    from scipy.optimize import newton
    root1_scipy = newton(func1, 1, fprime=grad1)
    root2_scipy = newton(func1, -1, fprime=grad1)

    root1_scipy_2 = newton(func2, 0, fprime=grad2)
    root2_scipy_2 = newton(func2, -1, fprime=grad2)
    root3_scipy_2 = newton(func2, 2, fprime=grad2)

    print("1st root found by Newton Method for function 1: {:0.8f}. Iterations: {}".format(root1_newton, iterations_1))
    print("1st root found by SciPy for function 1: {:0.8f}".format(root1_scipy))
    print("2nd root found by Newton Method for function 1: {:0.8f}. Iterations: {}".format(root2_newton, iterations_2))
    print("2nd root found by SciPy for function 1: {:0.8f}".format(root2_scipy))
    print("1st root found by Newton Method for function 2: {:0.8f}. Iterations: {}".format(root1_newton_2, iterations_3))
    print("1st root found by SciPy for function 2: {:0.8f}".format(root1_scipy_2))
    print("2nd root found by Newton Method for function 2: {:0.8f}. Iterations: {}".format(root2_newton_2, iterations_4))
    print("2nd root found by SciPy for function 2: {:0.8f}".format(root2_scipy_2))
    print("3rd root found by Newton Method for function 2: {:0.8f}. Iterations: {}".format(root3_newton_2, iterations_5))
    print("3rd root found by SciPy for function 2: {:0.8f}".format(root3_scipy_2))