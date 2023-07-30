import numpy as np
import matplotlib.pyplot as plt

# Define the simulation parameters
dt = 0.01      # Time step
total_time = 10.0   # Total simulation time
num_steps = int(total_time / dt)  # Number of simulation steps

# Constants
m = 1.0  # Mass of the body
g = 9.81  # Acceleration due to gravity
e3 = np.array([0, 0, 1])  # Gravity direction vector

# Inertia matrix (diagonal since expressed in the principal inertial frame)
Ixx, Iyy, Izz = 1.0, 2.0, 3.0
I = np.diag([Ixx, Iyy, Izz])

# Initial conditions
p0 = np.zeros(3,dtype=float) # Initial position
R0 = np.eye(3,dtype=float) # Initial orientation (identity matrix)
v0 = np.zeros(3,dtype=float) # Initial linear velocity
omega0 = np.zeros(3,dtype=float) # Initial angular velocity

# Lists to store the trajectory
positions = [p0]
orientations = [R0] 
angles = np.zeros((num_steps + 1, 3))  # Array to store Euler angles

# Euler's method simulation
p, R, v, omega = p0, R0, v0, omega0

for i in range(num_steps):

    # Compute the forces and torques
    f = np.ones(3,dtype=float)*0
    tau =  np.ones(3,dtype=float)*0
    
    omega_skew = np.array([[0, -omega[2], omega[1]],
                           [omega[2], 0, -omega[0]],
                           [-omega[1], omega[0], 0]])
    
    # Update the linear and angular velocities
    v += dt * ( - g*e3 + np.dot(R, f)/m) 
    omega += dt * ( - omega_skew @ (I @ omega) + tau)
    
    # Update the position and orientation
    p += dt * v 
    R += dt * np.dot(R, omega_skew) 

    # Append current position and orientation to the trajectory lists
    positions.append(p.copy())
    orientations.append(R.copy())

    # Calculate Euler angles from the current orientation matrix R
    angles[i] = np.array([np.arctan2(R[2, 1], R[2, 2]), 
                          np.arcsin(-R[2, 0]), 
                          np.arctan2(R[1, 0], R[0, 0])])
    

# Convert the lists to numpy arrays for easier plotting
positions = np.array(positions)
orientations = np.array(orientations)

# Plot the trajectories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot position trajectory
ax1.plot(positions[:, 0], positions[:, 1], label='Position')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Position Trajectory')
ax1.legend()

# Plot orientation trajectory (Euler angles)
time_array = np.arange(num_steps + 1) * dt  # Create the time array
angles = np.degrees(angles)  # Convert angles to degrees
for i in range(3):
    ax2.plot(time_array, angles[:, i], label=f'Euler Angle {i+1}')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Euler Angles (degrees)')
ax2.set_title('Orientation Trajectory')
ax2.legend()

plt.tight_layout()
plt.show()