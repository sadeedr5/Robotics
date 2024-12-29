import numpy as np
import matplotlib.pyplot as plt

def distance_point_to_line(point, line):
    # Calculate distance of a point to a line (Ax + By + C = 0)
    x, y = point
    A, B, C = line
    return abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)

def fit_line(points):
    # Fit a line Ax + By + C = 0 to a set of points
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return -m, 1, -b  # Return line in the form Ax + By + C = 0

def split_and_merge(points, threshold=0.1):
    def recursive_split(points):
        if len(points) < 2:
            return []
        
        line = fit_line(points)
        distances = [distance_point_to_line(p, line) for p in points]
        max_dist = max(distances)
        
        if max_dist > threshold:
            split_index = distances.index(max_dist)
            return (recursive_split(points[:split_index + 1]) +
                    recursive_split(points[split_index:]))
        else:
            return [points]
    
    segments = recursive_split(points)
    # Optionally merge segments with similar lines
    return segments

# Example usage:
lidar_points = [(1.1, 3.35), (2, 2), (2.9, 0.65), (4, -1)]
segments = split_and_merge(lidar_points)
print(f"Extracted line segments: {segments}")
