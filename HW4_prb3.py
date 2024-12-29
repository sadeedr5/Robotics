import numpy as np

def least_squares_line_fit(points):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # Calculate slope (m) and intercept (b)
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

# Example usage:
lidar_points = [(1.1, 3.35), (2, 2), (2.9, 0.65), (4, -1)]
m, b = least_squares_line_fit(lidar_points)
print(f"Slope (m): {m}, Intercept (b): {b}")
