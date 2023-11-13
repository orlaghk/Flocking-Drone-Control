import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

# Simulation parameters
v0           = 1.0      # velocity
eta          = 0.5      # random fluctuation in angle (in radians)
L            = 10       # size of box
R            = 1        # interaction radius
Rsq          = R**2     # square of the interaction radius
dt           = 0.2      # time step
Nt           = 80       # number of time steps
N            = 1000     # number of birds

#np.random.seed(17)      # set the random number generator seed

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
    x += vx*dt
    y += vy*dt
    
    # apply boundary conditions
    x, y = apply_boundary_conditions(x,y,L)
    return x, y

def get_mean_theta(x,y,theta,Rsq):
    '''
    Compute the local average direction in a circle of radius R around
    each bird
    '''
    mean_theta = theta
    for bird in range(N):
        neighbors = (x-x[bird])**2+(y-y[bird])**2 < R**2
        sum_x = np.sum(np.cos(theta[neighbors]))
        sum_y = np.sum(np.sin(theta[neighbors]))
        mean_theta[bird] = np.arctan2(sum_y, sum_x)
    
    return mean_theta

def update_theta(x,y,theta,Rsq,eta,N):
    '''
    Update theta to be the mean value plus a random amount between
    -eta/2 and eta/2
    '''
    mean_theta = get_mean_theta(x,y,theta,Rsq)
    theta = mean_theta + eta*(np.random.rand(N,1)-0.5)
    return theta

def update_velocities(vx,vy,theta):
    '''
    Update the velocities given theta
    '''
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    return vx, vy

def step(x,y,vx,vy,theta,Rsq,eta,N,dt):
    '''
    Compute a step in the dynamics:
    - update the positions
    - compute the new velocities
    '''
    x,y = update_positions(x,y,vx,vy,dt,L)
    theta = update_theta(x,y,theta,Rsq,eta,N)
    vx,vy = update_velocities(vx,vy,theta)
    return x, y, vx, vy

def update_quiver(q,x,y,vx,vy):
    '''
    Update a quiver with new position and velocity information
    This is only used for plotting
    '''
    q.set_offsets(np.column_stack([x,y]))
    q.set_UVC(vx,vy)
    return q

# set up a figure
fig, ax = plt.subplots(figsize = (10,10))

# get the initial configuration
x, y, vx, vy, theta = initialize_birds(N,L)

# do an initial plot and set up the axes
q = plt.quiver(x,y,vx,vy)
ax.set(xlim=(0, L), ylim=(0, L))
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# do each step, updating the quiver and plotting the new one
for iT in range(Nt):
    x,y,vx,vy = step(x,y,vx,vy,theta,Rsq,eta,N,dt)
    q = update_quiver(q,x,y,vx,vy)
    clear_output(wait=True)
    display(fig)