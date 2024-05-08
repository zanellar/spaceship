from mpl_toolkits.mplot3d import art3d 
import numpy as np 
from matplotlib.patches import Circle, Rectangle 
import matplotlib.pyplot as plt

from spaceship.utils.ops import normal2rotmat

#https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
 

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
     
    R = normal2rotmat(normal) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(R, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
 
 
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

class Drone3DStopMotion:

    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    def __init__(self, skipframes=0, lowerlimits=None, upperlimits=None):
        """
        Prepares a 3D scene 
        @ lowerlimits: list of lower limits for the axes
        @ upperlimits: list of upper limits for the axes
        """
        self.skipframes = skipframes
        
        self.fig = plt.figure(figsize=(10, 10))

        self.ax = plt.axes(projection = '3d') #Create axes

        self.lowerlimits = lowerlimits
        self.upperlimits = upperlimits
          
        # Plot the original reference frame
        self.ax.quiver(*self.origin, *self.x_axis, color='k', label='X-axis', alpha=0.2)
        self.ax.quiver(*self.origin, *self.y_axis, color='k', label='Y-axis', alpha=0.2)
        self.ax.quiver(*self.origin, *self.z_axis, color='k', label='Z-axis', alpha=0.2)

        # Make the axes equally spaced
        self.ax.set_box_aspect([1.0, 1.0, 1.0])
        set_axes_equal(self.ax)
    
    def add_slits(self,slits, color='g', alpha=0.5):
        """
        Adds slits to the scene
        @ slits: list of dicts with keys "height", "width", "position"
        """

        for slit in slits: 
            rect = Rectangle((0, 0), slit["width"], slit["height"], facecolor=color, alpha=alpha)
            self.ax.add_patch(rect)
            x_pos = slit["position"][0] - slit["width"]/2
            y_pos = slit["position"][1]
            z_pos = slit["position"][2] + slit["height"]/2
            pathpatch_2d_to_3d(rect, z=z_pos, normal='y')
            pathpatch_translate(rect, delta=[x_pos, y_pos, z_pos])

    def add_drone(self, orientations, positions, color_disk='b', alpha_disk=0.5, color_normal='r', alpha_normal=0.5):
        """
        Adds a drone to the scene 
        @ orientations: list of orientations (normal vectors)
        @ positions: list of positions (3D vectors)
        """ 

        orientations = np.asarray(orientations).reshape(-1,3)
        positions = np.asarray(positions).reshape(-1,3)

        if orientations.shape[0] != positions.shape[0]:
            raise ValueError("orientations and positions must have the same length")
        n = positions.shape[0]
        
        margin = 0.5 
        self.axlim = [np.min(positions)+margin, np.max(positions)+margin]  

        for i,normal,position in zip(range(n),orientations,positions):

            if i % (self.skipframes+1) != 0:
                continue

            # Add a circle to the plot
            circle = Circle((0,0), .2, facecolor = color_disk, alpha = alpha_disk)
            self.ax.add_patch(circle)
            pathpatch_2d_to_3d(circle, z = 0, normal = normal)
            pathpatch_translate(circle, delta=position)
 
            # Draw the normal vector 
            center = circle._segment3d.mean(axis=0) # Calculate the mean of the vertices
            normal /= np.linalg.norm(normal)  # Make sure the vector is normalized
            length = 0.3 # Length of the normal arrow
            end_point = center + length * normal
            self.ax.plot3D([center[0], end_point[0]], [center[1], end_point[1]], [center[2], end_point[2]], color=color_normal, alpha=alpha_normal)

    def _set_limits(self): 
        if self.lowerlimits is not None and self.upperlimits is not None:  
            if type(self.lowerlimits) == list and type(self.upperlimits) == list:
                self.ax.set_xlim3d(self.lowerlimits[0], self.upperlimits[0])
                self.ax.set_ylim3d(self.lowerlimits[1], self.upperlimits[1])
                self.ax.set_zlim3d(self.lowerlimits[2], self.upperlimits[2])
            elif type(self.lowerlimits) in [int, float] and type(self.upperlimits) in [int, float]:
                self.ax.set_xlim3d(self.lowerlimits, self.upperlimits)
                self.ax.set_ylim3d(self.lowerlimits, self.upperlimits)
                self.ax.set_zlim3d(self.lowerlimits, self.upperlimits)
            else:
                raise ValueError("lowerlimits and upperlimits must be lists or numbers")
        
    def viz(self, show=True, save=False, path=None, camera=None): 

        # Set the limits of the axes
        self._set_limits() 

        # Make the axes equally spaced
        self.ax.set_box_aspect([1.0, 1.0, 1.0])
        set_axes_equal(self.ax)

        plt.tight_layout()
 
        # change the camera angle
        if camera is not None:
            self.ax.view_init(*camera)
        else:
            self.ax.view_init(30, 30)

        if save:
            plt.savefig(path)
        if show:
            plt.show()



############################################################################
############################################################################
############################################################################


if __name__ == '__main__':

    visualizer = Drone3DStopMotion()

    slits = [
        dict(height=3, width=0.2, position=(0, 1.2)),
        dict(height=3, width=0.2, position=(0, 1.6)) 
             ]
    
    visualizer.add_slits(slits)
  
    # rotations = [(0,0,1), (0.2,0,0.8), (0.5,0,0.5), (0.8,0,0.2), (0.8,0,-0.2)]
    # translations = [(0,0,0), (0,0,0.5), (0,0.5,1), (0,1,1), (0,1.5,1)]

    rotations = [ (0.8,0,0.2), (0.8,0,-0.2)]
    translations = [ (0,1,1), (0,1.5,1)]

    visualizer.add_drone(rotations, translations)

    visualizer.show()