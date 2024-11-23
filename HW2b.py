import numpy as np
import matplotlib.pyplot as plt

# Define four basis functions and their derivatives
def basis_functions(t):
    return np.array([1, t, t**2, t**3])

def basis_derivatives(t):
    return np.array([0, 1, 2*t, 3*t**2])

# Define six basis functions and their derivatives
def basis_functions_six(t):
    return np.array([1, t, t**2, t**3, t**4, t**5])

def basis_derivatives_six(t):
    return np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])

# Create matrix equation solver for either four or six basis functions
def create_matrix_eq(t0, tT, z0, zT, dz0, dzT, basis_func, basis_deriv_func):
    # Basis function values at t = 0 and t = T
    psi_0 = basis_func(t0)
    psi_T = basis_func(tT)
    dpsi_0 = basis_deriv_func(t0)
    dpsi_T = basis_deriv_func(tT)

    # Assemble matrix A and vector B for boundary conditions
    A = np.vstack([psi_0, dpsi_0, psi_T, dpsi_T])
    B = np.array([z0, dz0, zT, dzT])

    # Solve for the coefficients
    coeffs = np.linalg.solve(A, B)
    return coeffs

# Trajectory generation function
def trajectory(coeffs, t_vals, basis_func):
    basis_vals = np.array([basis_func(t) for t in t_vals])
    return basis_vals @ coeffs

# Define the initial and final conditions for both cases
# Case (i) conditions
t0_i, tT_i = 0, 10
x1_0_i, x1_T_i = 1, 5
x2_0_i, x2_T_i = 0, 5
x3_0_i, x3_T_i = -3, 5
x_0_i, x_T_i = 1, 1

# Case (ii) conditions
t0_ii, tT_ii = 0, 15
x1_0_ii, x1_T_ii = 1, 10
x2_0_ii, x2_T_ii = 2, 10
x3_0_ii, x3_T_ii = 1, 5
x_0_ii, x_T_ii = 1, 1

# Time values for plotting
t_vals_i = np.linspace(t0_i, tT_i, 100)
t_vals_ii = np.linspace(t0_ii, tT_ii, 100)

# Calculate coefficients for each case using four basis functions
x1_coeffs_i = create_matrix_eq(t0_i, tT_i, x1_0_i, x1_T_i, 0, 0, basis_functions, basis_derivatives)
x2_coeffs_i = create_matrix_eq(t0_i, tT_i, x2_0_i, x2_T_i, 0, 0, basis_functions, basis_derivatives)
x3_coeffs_i = create_matrix_eq(t0_i, tT_i, x3_0_i, x3_T_i, 0, 0, basis_functions, basis_derivatives)

x1_coeffs_ii = create_matrix_eq(t0_ii, tT_ii, x1_0_ii, x1_T_ii, 0, 0, basis_functions, basis_derivatives)
x2_coeffs_ii = create_matrix_eq(t0_ii, tT_ii, x2_0_ii, x2_T_ii, 0, 0, basis_functions, basis_derivatives)
x3_coeffs_ii = create_matrix_eq(t0_ii, tT_ii, x3_0_ii, x3_T_ii, 0, 0, basis_functions, basis_derivatives)

# Generate trajectories for both cases
x1_vals_i = trajectory(x1_coeffs_i, t_vals_i, basis_functions)
x2_vals_i = trajectory(x2_coeffs_i, t_vals_i, basis_functions)
x3_vals_i = trajectory(x3_coeffs_i, t_vals_i, basis_functions)

x1_vals_ii = trajectory(x1_coeffs_ii, t_vals_ii, basis_functions)
x2_vals_ii = trajectory(x2_coeffs_ii, t_vals_ii, basis_functions)
x3_vals_ii = trajectory(x3_coeffs_ii, t_vals_ii, basis_functions)

# Plot trajectories for Case (i)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t_vals_i, x1_vals_i, label="x1(t) - Case (i)")
plt.xlabel("Time (s)")
plt.ylabel("x1(t)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_vals_i, x2_vals_i, label="x2(t) - Case (i)")
plt.xlabel("Time (s)")
plt.ylabel("x2(t)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_vals_i, x3_vals_i, label="x3(t) - Case (i)")
plt.xlabel("Time (s)")
plt.ylabel("x3(t)")
plt.legend()
plt.grid()

plt.suptitle("Trajectories for x1, x2, x3 (Case i: tf=10)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot trajectories for Case (ii)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t_vals_ii, x1_vals_ii, label="x1(t) - Case (ii)")
plt.xlabel("Time (s)")
plt.ylabel("x1(t)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_vals_ii, x2_vals_ii, label="x2(t) - Case (ii)")
plt.xlabel("Time (s)")
plt.ylabel("x2(t)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_vals_ii, x3_vals_ii, label="x3(t) - Case (ii)")
plt.xlabel("Time (s)")
plt.ylabel("x3(t)")
plt.legend()
plt.grid()

plt.suptitle("Trajectories for x1, x2, x3 (Case ii: tf=15)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
