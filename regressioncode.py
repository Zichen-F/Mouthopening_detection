from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the Beta Exponential PDF with constraint f(1; C, a, b) = 1
def beta_exponential_pdf_with_constraint(x, a, b):
    """
    Beta Exponential Function with constraint that the curve passes through (1, 1).
    Args:
        x (array-like): Input data (x > 0).
        a (float): Shape parameter a.
        b (float): Shape parameter b.
    Returns:
        array-like: PDF values.
    """
    C = 1 / (np.exp(-a * 1) * (1 - np.exp(-1))**(b - 1))  # Ensure f(1) = 1
    return C * np.exp(-a * x) * (1 - np.exp(-x))**(b - 1)

# List of Excel files
excel_files = [
    "D:/Master/Cough project/cough_images/cough_40hz_20000/output_excel/processed/cough2_processed.xlsx",
    "D:/Master/Cough project/cough_images/cough_40hz_20000/output_excel/processed/cough3_processed.xlsx",
    "D:/Master/Cough project/cough_images/cough_40hz_20000/output_excel/processed/cough4_processed.xlsx",
    "D:/Master/Cough project/cough_images/cough_40hz_20000/output_excel/processed/cough5_processed.xlsx",
    "D:/Master/Cough project/cough_images/cough_40hz_20000/output_excel/processed/cough6_processed.xlsx",
    "D:/Master/Cough project/cough_images/cough_40hz_20000/output_excel/processed/cough7_processed.xlsx"
]

# Define the x cutoff value
x_cutoff = 8  # Change this value to your desired cutoff

# Combine data with cutoff
all_t_data = []
all_y_data = []
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'black']  # Assign colors for each file

plt.figure(figsize=(10, 6))

for i, file in enumerate(excel_files):
    df = pd.read_excel(file)
    t_data = df['t_tmax'].values
    y_data = df['normalized_data'].values

    # Apply cutoff and filter valid data
    valid_indices = (np.isfinite(t_data) & np.isfinite(y_data) & (t_data <= x_cutoff))
    t_filtered = t_data[valid_indices]
    y_filtered = y_data[valid_indices]

    # Sort data to ensure a continuous line
    sorted_indices = np.argsort(t_filtered)
    t_sorted = t_filtered[sorted_indices]
    y_sorted = y_filtered[sorted_indices]

    # Plot each file's data as a line instead of scatter
    plt.plot(t_sorted, y_sorted, linestyle='-', color=colors[i], alpha=0.8, label=f'Cough {i+2}')  # Added label for each cough

    # Store combined data for fitting
    all_t_data.extend(t_sorted)
    all_y_data.extend(y_sorted)

# Convert combined data to numpy arrays
all_t_data = np.array(all_t_data)
all_y_data = np.array(all_y_data)

# Fit Beta Exponential distribution to the filtered dataset
try:
    initial_guess = [0, 2]
    bounds = ([0, 0], [np.inf, np.inf])  # Ensure positive values for a and b
    popt, _ = curve_fit(
        lambda x, a, b: beta_exponential_pdf_with_constraint(x, a, b),  # Fitting function
        all_t_data, all_y_data, p0=initial_guess, bounds=bounds
    )
except Exception as e:
    print(f"Error fitting Beta Exponential distribution: {e}")
    popt = [None, None]

# Generate fitted Beta Exponential curve
if all(param is not None for param in popt):
    t_fit = np.linspace(min(all_t_data), max(all_t_data), 200)
    y_fit_curve = beta_exponential_pdf_with_constraint(t_fit, *popt)
else:
    t_fit = np.linspace(min(all_t_data), max(all_t_data), 200)
    y_fit_curve = np.full_like(t_fit, np.nan)

# Compute error metrics
if not np.isnan(y_fit_curve).any():
    mse = np.mean((all_y_data - beta_exponential_pdf_with_constraint(all_t_data, *popt)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = np.mean(np.abs(all_y_data - beta_exponential_pdf_with_constraint(all_t_data, *popt)))  # Mean Absolute Error
else:
    mse, rmse, mae = np.nan, np.nan, np.nan

# Plot the fitted Beta Exponential curve
plt.plot(t_fit, y_fit_curve, linestyle='--', color='black', linewidth=2, label='Fitted Curve')

# Add labels, title, and legend
plt.xlim(0, 8)
plt.ylim(0, 1.1)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\tilde{A}$')
plt.title('Normalized Cough Data with Fitted Curve')
plt.legend(loc='upper right')  # Added legend with specific location
plt.grid()
plt.show()

# Print fit parameters and errors
if all(param is not None for param in popt):
    print("\nBeta Exponential Fit Parameters (Constrained):")
    print(f"a = {popt[0]:.2f}, b = {popt[1]:.2f}")
else:
    print("Fit failed. Parameters could not be estimated.")

print("\nError Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
