import numpy as np
import matplotlib.pyplot as plt


def cubic_spline_interpolation(x_vals, y_vals, q0, qn):
    n = len(x_vals)
    h = np.diff(x_vals)
    b = np.diff(y_vals) / h

    # Formulating the system of equations
    A = np.zeros((n, n))
    rhs = np.zeros(n)

    # Boundary conditions
    A[0, 0] = 1
    A[-1, -1] = 1
    rhs[0] = q0
    rhs[-1] = qn

    # Fill the A matrix and rhs vector for the interior points
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * (b[i] - b[i - 1])

    # Solve the system
    m = np.linalg.solve(A, rhs)

    # Coefficients of the cubic spline
    def spline(x, i):
        t = (x - x_vals[i]) / h[i]
        return (1 - t) * y_vals[i] + t * y_vals[i + 1] + t * (1 - t) * (
                (1 - t) * (m[i] * (h[i] / 3)) + t * (m[i + 1] * (h[i] / 3)))

    # Function to evaluate spline at any x
    def evaluate_spline(x):
        for i in range(n - 1):
            if x_vals[i] <= x <= x_vals[i + 1]:
                return spline(x, i)
        raise ValueError("The input x is out of the interpolation range.")
    print(evaluate_spline(0.8))
    return evaluate_spline


x_vals = [0.1, 0.5, 0.9, 1.3, 1.7]
y_vals = [100, 4, 1.2346, 0.59172, 0.34602]
q0 = 0
qn = 0

spline_func = cubic_spline_interpolation(x_vals, y_vals, q0, qn)

x_plot = np.linspace(min(x_vals), max(x_vals), 500)
y_plot = [spline_func(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'ro', label='Data points')
plt.plot(x_plot, y_plot, 'b-', label='Cubic Spline')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()
