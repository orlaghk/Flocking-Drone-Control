import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

# Simulation parameters
vmax = 1.0       # maximum velocity
lambda_c = 0.1   # centering parameter
lambda_a = 0.1   # avoidance parameter
lambda_m = 0.1   # matching parameter
R = 1.0          # interaction radius
r = 0.1          # minimum distance
dt = 0.2         # time step
Nt = 80          # number of time steps
N = 100          # number of birds
L = 10           # size of box

def initialize_birds(N, L):
    x = np.random.rand(N, 1) * L
    y = np.random.rand(N, 1) * L
    theta = 2 * np.pi * np.random.rand(N, 1)
    vx = vmax * np.cos(theta)
    vy = vmax * np.sin(theta)
    return x, y, vx, vy, theta

def apply_boundary_conditions(x, y, L):
    x = x % L
    y = y % L
    return x, y

def update_positions(x, y, vx, vy, dt, L):
    x += vx * dt
    y += vy * dt
    x, y = apply_boundary_conditions(x, y, L)
    return x, y

def get_mean_theta(x, y, theta, R):
    mean_theta = theta.copy()
    for bird in range(N):
        neighbors = np.linalg.norm(np.column_stack([x - x[bird], y - y[bird]]), axis=1) < R
        sum_theta = np.sum(theta[neighbors])
        mean_theta[bird] = sum_theta / np.count_nonzero(neighbors)
    return mean_theta

def update_theta(x, y, theta, R, eta):
    mean_theta = get_mean_theta(x, y, theta, R)
    theta += eta * (np.random.rand(N, 1) - 0.5)
    theta = mean_theta + eta * (np.random.rand(N, 1) - 0.5)
    return theta

def update_velocities(vx, vy, theta):
    vx = vmax * np.cos(theta)
    vy = vmax * np.sin(theta)
    return vx, vy

def step(x, y, vx, vy, theta, dt, L, R, r, lambda_c, lambda_a, lambda_m):
    x, y = update_positions(x, y, vx, vy, dt, L)
    mean_theta = get_mean_theta(x, y, theta, R)
    v_c = lambda_c * (np.column_stack([np.cos(mean_theta), np.sin(mean_theta)]) - np.column_stack([x, y]))
    too_close = np.linalg.norm(np.column_stack([x, y]) - np.column_stack([x, y]), axis=1) < r
    avoidance = lambda_a * np.sum(np.column_stack([x, y]) - np.column_stack([x, y])[too_close], axis=0)
    avg_velocity = np.mean(np.column_stack([vx, vy]), axis=0)
    v_m = lambda_m * (avg_velocity - np.column_stack([vx, vy]))
    new_v = vx + v_c[:, 0] + avoidance[0] + v_m[:, 0], vy + v_c[:, 1] + avoidance[1] + v_m[:, 1]
    new_v_magnitude = np.linalg.norm(np.column_stack(new_v), axis=1)
    new_v = np.column_stack([v / max(1, v_magnitude) for v, v_magnitude in zip(np.column_stack(new_v), new_v_magnitude)])
    x, y = update_positions(x, y, new_v[:, 0], new_v[:, 1], dt, L)
    return x, y, new_v[:, 0], new_v[:, 1]

def update_quiver(q, x, y, vx, vy):
    q.set_offsets(np.column_stack([x, y]))
    q.set_UVC(vx, vy)
    return q

# set up a figure
fig, ax = plt.subplots(figsize=(10, 10))

# get the initial configuration
x, y, vx, vy, theta = initialize_birds(N, L)

# do an initial plot and set up the axes
q = plt.quiver(x, y, vx, vy)
ax.set(xlim=(0, L), ylim=(0, L))
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# do each step, updating the quiver and plotting the new one
for iT in range(Nt):
    x, y, vx, vy = step(x, y, vx, vy, theta, dt, L, R, r, lambda_c, lambda_a, lambda_m)
    q = update_quiver(q, x, y, vx, vy)
    clear_output(wait=True)
    display(fig)
