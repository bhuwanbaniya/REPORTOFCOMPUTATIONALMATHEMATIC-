import numpy as np
import matplotlib.pyplot as plt
#BHUWANBANIYA
#2414002
def plot_function(func, a, b):
    """
    This function plot the graph of the input func
    within the given interval [a,b).
    """
    x = np.linspace(a, b, 100)
    y = func(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graph of the function')
    plt.grid(True)
    plt.show()

def trapezoidal_approx(func, a, b, N):
    '''Compute the Trapezoidal Approximation of Definite Integral of a function over the interval [a,b].

    Parameters
    ----------
    func : function
           Vectorized function of one variable
    a , b : numbers
        Endpoints of the interval [a,b]
    N : integer
        Number of subintervals of equal length in the partition of [a,b]

    Returns
    -------
    float
        Approximation of the definite integral by Trapezoidal Approximation.
    '''

    dx = (b - a) / N  # Step size
    x = np.linspace(a, b, N+1)  # Equidistant partition points
    return 0.5*dx*np.sum(func(x[:-1]) + func(x[1:]))  # Trapezoidal Approximation of Definite Integral

if _name_ == "_main_":
    # 1st Function to be integrated
    func_1 = lambda x : x/(x**2 + 1)
    # Indefinite Integral of the function
    antiderivative_1 = lambda x: np.log(1 + x**2)  # Indefinite Integral of the function

    # 2nd Function to be integrated
    func_2 = lambda x : np.exp(x)
    # Indefinite Integral of the function
    antiderivative_2 = lambda x: np.exp(x)  # Indefinite Integral of the function

    # End points for 1st Function
    a1 = 0; b1 = 5
    # End points for 2nd Function
    a2 = 0; b2 = 5

    # Call the function to Plot the graph of the functions
    plot_function(func_1, a1, b1)
    plot_function(func_2, a2, b2)

    # Number of partitions for 1st Function
    N1 = 500  # Change the value as required
    # Number of partitions for 2nd Function
    N2 = 500  # Change the value as required

    # Call trapezoidal_approx to compute Trapezoidal Approximation:
    trapezoidal_approx_1 = trapezoidal_approx(func_1, a1, b1, N1)
    trapezoidal_approx_2 = trapezoidal_approx(func_2, a2, b2, N2)

    # Calculate the true value of the definite integral
    definite_integral_1 = antiderivative_1(b1) - antiderivative_1(a1)  # For 1st Function
    definite_integral_2 = antiderivative_2(b2) - antiderivative_2(a2)  # For 2nd Function

    # Calculate the relative error between the approximate value and true value
    relative_error_1 = ((trapezoidal_approx_1 - definite_integral_1) / trapezoidal_approx_1) * 100  # For 1st Function
    relative_error_2 = ((trapezoidal_approx_2 - definite_integral_2) / trapezoidal_approx_2) * 100  # For 2nd Function

    print("Subinterval width = {:0.6f}".format((b1-a1)/N1))
    print("Trapezoidal Approximation for 1st Function = {:0.6f}".format(trapezoidal_approx_1))
    print("Actual Value for 1st Function = {:0.6f}".format(definite_integral_1))
    print("Relative error for 1st Function ={:0.8f}%".format(relative_error_1))

    print("Trapezoidal Approximation for 2nd Function = {:0.6f}".format(trapezoidal_approx_2))
    print("Actual Value for 2nd Function = {:0.6f}".format(definite_integral_2))
    print("Relative error for 2nd Function ={:0.8f}%".format(relative_error_2))