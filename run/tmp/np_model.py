import numpy as np
import matplotlib.pyplot as plt
from spaceship.utils.ops import normal2rotmat, rotmat2euler 

# Define the simulation parameters
dt = 0.01      # Time step
total_time = 1.0   # Total simulation time
num_steps = int(total_time / dt)  # Number of simulation steps

# Constants
m = 1.0  # Mass of the body
g = 9.81  # Acceleration due to gravity
e3 = np.array([0, 0, 1])  # Gravity direction vector

# Inertia matrix (diagonal since expressed in the principal inertial frame)
Jxx, Jyy, Jzz = 1.0, 1.0, 1.0
J = np.diag([Jxx, Jyy, Jzz])

# Initial conditions
p0 = np.array([0,0,0],dtype=float) # Initial position
n0 = np.array([0,0,1],dtype=float) # Initial orientation (identity matrix)
v0 = np.array([0,1,0],dtype=float) # Initial linear velocity
omega0 = np.array([0.5,0,0.5],dtype=float) # Initial angular velocity

# Lists to store the trajectory
positions = [p0]
orientations = [n0] 
angles = [rotmat2euler(normal2rotmat(n0))]   

# Euler's method simulation
p, n, v, omega = p0, n0, v0, omega0

# Compute the forces and torques
def f1(t):
    return np.array([0, 0, 0])

def tau1(t):
    return np.array([0, 0, 0])

def J1(t):
    return np.diag([0, 0, 0])

f = np.zeros(3,dtype=float) 
tau =  np.zeros(3,dtype=float) 

for t in range(num_steps):

    omega_skew = np.array([[0, -omega[2], omega[1]],
                           [omega[2], 0, -omega[0]],
                           [-omega[1], omega[0], 0]])
    
    # get rotation matrix from normal vector
    R = normal2rotmat(n)

    # Compute the forces and torques 
    f = m*g*R.T@e3 + f1(t) 
    tau = - omega_skew @ J1(t) @ omega + tau1(t)
    
    print(f" tau = {tau}, omega = {omega}")
    
    # Update the linear and angular velocities
    v += dt * ( - g*e3 +  R@f/m) 
    omega += dt * np.linalg.inv(J)@( - omega_skew @ J @ omega + tau)
    
    # Update the position and orientation
    p += dt * v 
    n += dt * np.dot(n, omega_skew) 

    # Append current position and orientation to the trajectory lists
    positions.append(p.copy())
    orientations.append(n.copy())

    # Calculate Euler angles from the current orientation matrix R
    angles.append(rotmat2euler(R))
    

# Convert the lists to numpy arrays for easier plotting
positions = np.array(positions)
orientations = np.array(orientations)

# Plot the trajectories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot position trajectory
ax1.plot(positions[:, 0], positions[:, 1], label='Position')
ax1.plot(positions[0, 0], positions[0, 1], 'o', label='Start')
ax1.plot(positions[-1, 0], positions[-1, 1], 'o', label='End')
ax1.axis('equal') 
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Position Trajectory')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 3)
ax1.legend()

# Plot orientation trajectory (Euler angles)
time_array = np.arange(num_steps+1) * dt  # Create the time array
angles = np.degrees(angles)  # Convert angles to degrees
for i in range(3):
    ax2.plot(time_array, angles[:, i], label=f'Euler Angle {i+1}')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Euler Angles (degrees)')
ax2.set_title('Orientation Trajectory') 
ax2.legend()

plt.tight_layout()
plt.show()

# Graphics
from spaceship.visual.draw3d import Drone3DStopMotion
pos_slit = np.array([-0.1, 1.2, 1.5] )
slits = [dict(height=3, width=0.2, position=pos_slit)]
graphics = Drone3DStopMotion(skipframes=2, lowerlimits=-1, upperlimits=3)
graphics.add_slits(slits)
graphics.add_drone(orientations, positions)
graphics.show()
  