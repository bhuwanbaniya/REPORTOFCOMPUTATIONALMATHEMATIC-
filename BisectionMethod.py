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
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of the function')
    plt.grid(True)
    plt.show()

def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find the root of a function within a given interval.

    Parameters:
    - func: The function for which the root is to be found.
    - a, b: Interval [a, b] within which the root is searched for.
    - tol: Tolerance level for checking convergence of the method.
    - max_iter: Maximum number of iterations.

    Returns:
    - root: Approximation of the root.
    - iterations: Number of iterations taken to find the root.
    """
    # Check if the interval is valid (signs of f(a) and f(b) are different)
    if np.sign(func(a)) * np.sign(func(b)) >= 0:
        raise ValueError("The signs of f(a) and f(b) must be different.")

    # Main loop starts here
    iter_count = 1
    while iter_count <= max_iter:
        c = (a + b) / 2  # Midpoint
        if func(c) == 0 or (b - a) / 2 < tol:
            return c, iter_count  # Found the root and number of iterations
        if np.sign(func(c)) * np.sign(func(a)) < 0:
            b = c  # Root is in the left half
        else:
            a = c  # Root is in the right half
        iter_count += 1

    print("Warning! Exceeded the maximum number of iterations.")
    return (a + b) / 2, max_iter  # Return the last approximation and max iterations

if _name_ == "_main_":
    # Define the first function for which the root is to be found
    func1 = lambda x: x**2 - x - 1  # First Function

    # Define the second function
    func2 = lambda x: x*3 - x*2 - 2*x + 1  # Second Function

    # Call plot_function to plot graph of the first function
    plot_function(func1, -1, 2)

    # Set the interval [a, b] for the search for the first function
    a_1 = 1; b_1 = 2  # For first root of the first function
    a_2 = -1; b_2 = -0.5  # For second root of the first function

    # Call the bisection method for the first function
    our_root_1, iterations_1 = bisection_method(func1, a_1, b_1)
    our_root_2, iterations_2 = bisection_method(func1, a_2, b_2)

    # Call SciPy method root for the first function
    sp_result_1 = sp.optimize.root(func1, (a_1 + b_1) / 2)
    sp_root_1 = sp_result_1.x.item()

    sp_result_2 = sp.optimize.root(func1, (a_2 + b_2) / 2)
    sp_root_2 = sp_result_2.x.item()

    # Print the result for the first function
    print("1st root found by Bisection Method for the first function = {:0.8f}. Iterations: {}".format(our_root_1, iterations_1))
    print("1st root found by SciPy for the first function = {:0.8f}".format(sp_root_1))

    print("2nd root found by Bisection Method for the first function = {:0.8f}. Iterations: {}".format(our_root_2, iterations_2))
    print("2nd root found by SciPy for the first function = {:0.8f}".format(sp_root_2))

    # Call plot_function to plot graph of the second function
    plot_function(func2, -1.5, 2)

    # Set the intervals [a, b] for the search for the second function
    a_3 = -1.5; b_3 = -1  # For first root of the second function
    a_4 = -0.5; b_4 = 0.5  # For second root of the second function
    a_5 = 1.5; b_5 = 2  # For third root of the second function

    # Call the bisection method for the second function
    our_root_3, iterations_3 = bisection_method(func2, a_3, b_3)
    our_root_4, iterations_4 = bisection_method(func2, a_4, b_4)
    our_root_5, iterations_5 = bisection_method(func2, a_5, b_5)

    # Call SciPy method root for the second function
    sp_result_3 = sp.optimize.root(func2, (a_3 + b_3) / 2)
    sp_root_3 = sp_result_3.x.item()

    sp_result_4 = sp.optimize.root(func2, (a_4 + b_4) / 2)
    sp_root_4 = sp_result_4.x.item()

    sp_result_5 = sp.optimize.root(func2, (a_5 + b_5) / 2)
    sp_root_5 = sp_result_5.x.item()

    # Print the result for the second function
    print("\n1st root found by Bisection Method for the second function = {:0.8f}. Iterations: {}".format(our_root_3, iterations_3))
    print("1st root found by SciPy for the second function = {:0.8f}".format(sp_root_3))

    print("2nd root found by Bisection Method for the second function = {:0.8f}. Iterations: {}".format(our_root_4, iterations_4))
    print("2nd root found by SciPy for the second function = {:0.8f}".format(sp_root_4))

    print("3rd root found by Bisection Method for the second function = {:0.8f}. Iterations: {}".format(our_root_5, iterations_5))
    print("3rd root found by SciPy for the second function = {:0.8f}".format(sp_root_5))