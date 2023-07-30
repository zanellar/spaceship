from mpl_toolkits.mplot3d import art3d 
import numpy as np 
from matplotlib.patches import Circle, Rectangle
from itertools import product 
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals


def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,    d[2],   -d[1]],
                        [-d[2], 0,      d[0]],
                        [d[1],  -d[0],  0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
 
 
def draw_normal(ax, center, normal, length=0.3):
    """
    Draws a normal vector as a line segment starting from the specified center.
    The normal vector is scaled by the specified length.
    """
    normal /= np.linalg.norm(normal)  # Make sure the vector is normalized
    end_point = center + length * normal
    ax.plot3D([center[0], end_point[0]], [center[1], end_point[1]], [center[2], end_point[2]], color='red', alpha=0.5)


def plot_slit(ax, height, width, position, color='g', alpha=0.5):
    """
    Add a rectangle to a 3D plot.

    Parameters:
        ax (Axes3D): The 3D axes object to which the rectangle will be added.
        height (float): The height of the rectangle.
        width (float): The width of the rectangle.
        position (tuple): The position (x, y, z) of the rectangle in 3D space.
        color (str, optional): The color of the rectangle. Default is 'g' (green).
        alpha (float, optional): The transparency of the rectangle. Default is 0.5.

    Returns:
        Rectangle: The rectangle patch object added to the 3D plot.
    """

    rect = Rectangle((0, 0), width, height, facecolor="g", alpha=0.5)
    ax.add_patch(rect)

    pathpatch_2d_to_3d(rect, z=position[2], normal='y')
    pathpatch_translate(rect, delta=position)

    return rect

def plot_drone(n,p):

    circle = Circle((0,0), .2, facecolor = 'b', alpha = .5)
    ax.add_patch(circle)
    pathpatch_2d_to_3d(circle, z = 0, normal = n)
    pathpatch_translate(circle, delta=p)

    # Get the center and normal of the circle in 3D
    center_3d = circle._segment3d.mean(axis=0)  # Calculate the mean of the vertices

    # Draw the normal vector as a red line segment
    draw_normal(ax, center_3d, n)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

############################################################################
############################################################################
############################################################################

ax = plt.axes(projection = '3d') #Create axes


############################################################################
# plot rectangle 

height = 3
width = 0.2
position = (-0.1, 1.2, 1.5) 

plot_slit(ax, height, width, position)

############################################################################
# plot circle

rotations = [(0,0,1), (0.2,0,0.8), (0.5,0,0.5), (0.8,0,0.2)]
translations = [(0,0,0), (0,0,0.5), (0,0.5,1), (0,1,1)]

for n,p in zip(rotations, translations):
    plot_drone(n,p)
    
 
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,3)
ax.set_zlim3d(-1,4)


ax.set_box_aspect([1.0, 1.0, 1.0])

set_axes_equal(ax)

plt.show()