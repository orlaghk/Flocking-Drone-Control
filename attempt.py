import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
N = 10  # Number of birds
L = 10  # Size of box
R = 2  # Range for observing other birds
r = 1  # Minimum distance to other birds
v_max = 10  # Maximum velocity
lambda_c = 1
lambda_a = 1
lambda_m = 1
dt = 0.2  # Time step
Nt = 100  # Number of time steps
v0 = 1.0      # velocity

def initialize_birds(N,L):
    '''
    Set initial positions, direction, and velocities 
    '''
    # bird positions
    x = np.random.rand(N,1)*L
    y = np.random.rand(N,1)*L

    # bird velocities
    theta = 2 * np.pi * np.random.rand(N,1)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    return x, y, vx, vy, theta

def apply_boundary_conditions(x,y,L):
    '''
    Apply periodic boundary conditions
    '''
    x = x % L
    y = y % L
    return x, y

def update_positions(x,y,xv,vy,dt,L):
    '''
    Update the positions moving dt in the direction of the velocity
    and applying the boundary conditions
    '''
    
    # update positions
    x += xv*dt
    y += vy*dt
    
    # apply boundary conditions
    x, y = apply_boundary_conditions(x,y,L)
    return x, y

# Initial positions, velocities, and directions
x, y, vx, vy, theta = initialize_birds(N, L)

# Function to find neighbors within range R
def get_local_birds(x, y, i):
    return {j: j for j in range(N) if np.linalg.norm([x[j] - x[i], y[j] - y[i]]) < R and i != j}

# Function to find birds that are too close (within distance r)
def get_too_close_birds(x, y, i):
    return {j: j for j in range(N) if np.linalg.norm([x[j] - x[i], y[j] - y[i]]) < r and i != j}

# Function to limit speed
def limit_speed(vx, vy, v_max):
    speed = np.linalg.norm([vx, vy])
    if speed > v_max:
        vx = (v_max / speed) * vx
        vy = (v_max / speed) * vy
    return vx, vy

# Function to update velocities based on flocking rules
def update_velocities(x, y, vx, vy):
    v_c = lambda_c * ((np.sum([x[j] for j in local_birds]) / len(local_birds) - x[i]) if len(local_birds) > 0 else 0,
                      (np.sum([y[j] for j in local_birds]) / len(local_birds) - y[i]) if len(local_birds) > 0 else 0)

    v_a = lambda_a * ((np.sum([x[i] - x[j] for j in too_close_birds]) if len(too_close_birds) > 0 else 0),
                      (np.sum([y[i] - y[j] for j in too_close_birds]) if len(too_close_birds) > 0 else 0))

    v_m = lambda_m * ((np.sum([vx[j] for j in local_birds]) / len(local_birds) - vx[i]) if len(local_birds) > 0 else 0,
                      (np.sum([vy[j] for j in local_birds]) / len(local_birds) - vy[i]) if len(local_birds) > 0 else 0)

    vx_i, vy_i = vx[i] + v_c[0] + v_a[0] + v_m[0], vy[i] + v_c[1] + v_a[1] + v_m[1]

    # Limit speed
    vx_i, vy_i = limit_speed(vx_i, vy_i, v_max)

    return vx_i, vy_i

# Run simulation
for _ in range(Nt):
    for i in range(N):
        local_birds = get_local_birds(x, y, i)
        too_close_birds = get_too_close_birds(x, y, i)

        vx[i], vy[i] = update_velocities(x, y, vx, vy)
        x[i], y[i] = update_positions(x[i], y[i], vx[i], vy[i], dt, L)

# Plotting
plt.figure(figsize=(8, 8))
plt.quiver(x, y, vx, vy, scale=10, color='blue', width=0.007)
plt.title('Bird Flocking Simulation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
