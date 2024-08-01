import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torus import Torus
from matplotlib.colors import LinearSegmentedColormap


def smooth_random_field(size):
    X, Y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))

    result = np.zeros((size, size))
    for i in range(5):  # Use 5 waves
        freq = 2 ** i
        phase_x, phase_y = np.random.random(2) * 2 * np.pi
        result += np.sin(freq * 2 * np.pi * X + phase_x) * np.sin(freq * 2 * np.pi * Y + phase_y)

    result = (result - result.min()) / (result.max() - result.min())
    return result


def create_smooth_torus(size):
    data = smooth_random_field(size)  # We're using option 3 by default here.
    return Torus(data)


def opposite_point(i, j, size):
    # First, find opposite-but-same-vertical-section
    opp_i = (i + size // 2) % size

    # Find how far we are from the "top" of our vertical section
    local_j = min((j - size // 2) % size, (size // 2 - j) % size)

    # They have same vertical-section-displacement, but opposite
    opp_j = (size // 2 - local_j) % size
    if j >= size // 2:
        opp_j = (size // 2 + local_j) % size

    return opp_i, opp_j


def add_holes(torus, holes):
    size = torus.shape[0]
    for center_i, center_j, radius in holes:
        for i in range(size):
            for j in range(size):
                di = min((i - center_i) % size, (center_i - i) % size)
                dj = min((j - center_j) % size, (center_j - j) % size)
                if di ** 2 + dj ** 2 < radius ** 2:
                    opp_i, opp_j = opposite_point(i, j, size)
                    torus[i, j] = -1  # Primary hole surface
                    torus[opp_i, opp_j] = -1  # Opposite hole surface
    return torus


def create_custom_colormap():
    colors = ['#8B4513', '#D2B48C', '#FFFF00', '#FFA500', '#FF0000', '#700000']
    return LinearSegmentedColormap.from_list("custom", colors)


def plot_torus(ax, torus, theta_offset=0, phi_offset=0):
    size = torus.shape[0]
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, size), np.linspace(0, 2 * np.pi, size))

    R, r = 1, 0.3  # Major and minor radii of the torus

    # Apply rotation around vertical and horizontal axes
    x = (R + r * np.cos(phi)) * np.cos(theta + theta_offset)
    y = (R + r * np.cos(phi)) * np.sin(theta + theta_offset)
    z = r * np.sin(phi)

    # Rotate around horizontal axis
    y_new = y * np.cos(phi_offset) - z * np.sin(phi_offset)
    z_new = y * np.sin(phi_offset) + z * np.cos(phi_offset)

    cmap = create_custom_colormap()
    cmap.set_bad('black', 1.0)

    masked_data = np.ma.masked_where(torus.arr == -1, torus.arr)

    ax.clear()
    ax.plot_surface(x, y_new, z_new, facecolors=cmap(masked_data), alpha=0.7)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_axis_off()


# Create three different tori with holes
size = 100

# Torus 1: Single large hole
torus1 = create_smooth_torus(size)
holes1 = [(50, 50, 14)]
torus1 = add_holes(torus1, holes1)
# Torus 2: Multiple small holes
torus2 = create_smooth_torus(size)
holes2 = [(25, 25, 6), (75, 65, 5), (15, 75, 4)]
torus2 = add_holes(torus2, holes2)
# Torus 3: Combination of large and small holes
torus3 = create_smooth_torus(size)
holes3 = [(20, 50, 12), (20, 10, 7), (90, 70, 5), (20, 90, 6)]
torus3 = add_holes(torus3, holes3)

# Create the plot
fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')


def update(frame):
    theta_offset = frame * np.pi / 180  # Rotate 1 degree per frame about vertical axis
    phi_offset = frame * np.pi / 1440  # Rotate 0.125 degrees per frame about horizontal axis

    plot_torus(ax1, torus1, theta_offset, phi_offset)
    ax1.set_title("Torus with Single Large Hole")

    plot_torus(ax2, torus2, theta_offset, phi_offset)
    ax2.set_title("Torus with Multiple Small Holes")

    plot_torus(ax3, torus3, theta_offset, phi_offset)
    ax3.set_title("Torus with Large and Small Holes")


ani = FuncAnimation(fig, update, frames=360, interval=100, blit=False)
plt.tight_layout()
plt.show()