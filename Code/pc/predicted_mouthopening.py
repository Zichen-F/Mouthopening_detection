import numpy as np
import matplotlib.pyplot as plt


# Constants
a = 2.17
b = 4.57
t_max = 0.32  # Example maximum time in seconds
a_max = 0.000329



# Normalization constant
C = 1 / (np.exp(-a * 1) * (1 - np.exp(-1)) ** (b - 1))

# Function definition
def f(t, t_max):
    x = t / t_max  # Normalize t to [0, 1]
    return C * np.exp(-a * x) * (1 - np.exp(-x)) ** (b - 1)

# Generate t values
t_values = np.linspace(0.1, 0.7, 600)
an_values = f(t_values, t_max)
a_values = an_values * a_max

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(t_values, a_values, label='Theoretical f(t)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('f(t)')
plt.title('Comparison of Theoretical f(t) and Shifted Data')
plt.legend()
plt.grid(True)
plt.show()
