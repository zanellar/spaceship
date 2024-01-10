import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy 

# Create rotation matrices
def rotation_matrix1(xn): 
    e3 = np.array([0,0,1]).reshape((3,1))
    xn = np.array(xn).reshape((3,1))
    norm_e3n = np.linalg.norm(e3 - xn)**2
    if norm_e3n == 0:
        R = np.eye(3)
    else:
        R =np.eye(3) + np.matmul(e3 - xn,np.transpose(e3 - xn)) / norm_e3n 
    return R

def cayley_transform(A):
    A=np.linalg.inv(np.eye(3)+A)@(np.eye(3)-A)
    return A

def skew_matrix(A): 
    return 0.5*(A-np.linalg.pinv(A))

def orthogonalize_cay(A):
    return cayley_transform(skew_matrix(cayley_transform(A)))

def orthogonalize_polar(A): 
    return A@np.linalg.pinv(scipy.linalg.sqrtm(np.transpose(A)@A))

def orthogonalize_svd(A):
    U, S, V = np.linalg.svd(A)
    return U@V

def orthogonalize_gram(A):
    return scipy.linalg.orth(A)

def orthogonalize_log(A):
    return scipy.linalg.expm(skew_matrix(scipy.linalg.logm(A)))
 

def rotation_matrix2(xn):  
    e3 = np.array([0.,0.,1.]) 
    e3xn = np.cross(e3, xn)
    skew_e3xn = np.array([[0, -e3xn[2], e3xn[1]],
                        [e3xn[2], 0, -e3xn[0]],
                        [-e3xn[1], e3xn[0], 0]]) 
    I_3 = np.eye(3)
    sin_theta = np.linalg.norm(e3xn) 
    cos_theta = np.dot(e3, xn)  
    R = I_3 + sin_theta*skew_e3xn + (1. - cos_theta)*np.matmul(skew_e3xn,skew_e3xn)
    R = orthogonalize_svd(R)
    return R

rotation_matrix = rotation_matrix2

# Define the reference frame
origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])
 
# Specify rotation angles in radians
normals = []  # Modify these angles as needed
normals.append([0, 0, 1])
normals.append([0, 0.50544946, 0.86285621])
normals.append([0, 0.70710678, 0.70710678])
normals.append([0, 0, -1])

for n in normals: 
    print(f"n = {np.linalg.norm(n)}")

# Create subplots for each angle
fig, axes = plt.subplots(1, len(normals), subplot_kw={'projection': '3d'}, figsize=(12, 4))

for i, xn in enumerate(normals):
    ax = axes[i]

    # Plot the original reference frame
    ax.quiver(*origin, *x_axis, color='k', label='X-axis', alpha=0.2)
    ax.quiver(*origin, *y_axis, color='k', label='Y-axis', alpha=0.2)
    ax.quiver(*origin, *z_axis, color='k', label='Z-axis', alpha=0.2)

    # Apply rotation matrix
    R = rotation_matrix(xn)
    new_x = np.dot(R, x_axis)
    new_y = np.dot(R, y_axis)
    new_z = np.dot(R, z_axis)
    print(f"theta = {np.arccos(np.dot(new_z,z_axis))*180/np.pi}")
    print(R)
    print(np.linalg.norm(new_x), np.linalg.norm(new_y), np.linalg.norm(new_z))
    ax.quiver(*origin, *new_x, color='r')
    ax.quiver(*origin, *new_y, color='g' )
    ax.quiver(*origin, *new_z, color='b' )

    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Rotation: {xn}')

    # Add legend
    ax.legend()

plt.tight_layout()
plt.show()