import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("tvecLog.txt") as fd:
    tvecs = list(map(lambda x: np.array([[float(i)] for i in x.split(",")]), fd.readlines()))

x_vals = [t[0, 0] for t in tvecs]
y_vals = [t[1, 0] for t in tvecs]
z_vals = [t[2, 0] for t in tvecs]

def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

x_smooth = moving_average(x_vals)
y_smooth = moving_average(y_vals)
z_smooth = moving_average(z_vals)

step = 20
x_downsampled = x_smooth[::step]
y_downsampled = y_smooth[::step]
z_downsampled = z_smooth[::step]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_vals, y_vals, z_vals, marker='o', color='gray', alpha=0.5, label="Original Points")

ax.plot(x_downsampled, y_downsampled, z_downsampled, marker='o', linestyle='-', color='b', label="Filtered Path")


ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Position")

plt.legend()
plt.show()

