import numpy as np  


def normal2rotmat(n):
    """
    Calculates a rotation matrix from a normal vector n.
    """ 
    sign = np.sign(np.dot(n, (0, 0, 1))) #Figure out which way is up
    sign = 1 if sign == 0 else sign 
    d = sign*np.cross(n, (0, 0, 1)) #Obtain the rotation vector   
    R = rotation_matrix(d) #Get the rotation matrix
    return R

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

    R = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return R


def rotmat2euler(R):
    """
    Calculates the Euler angles from a rotation matrix R.
    """
    phi = np.arctan2(R[2,1], R[2,2])
    theta = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    psi = np.arctan2(R[1,0], R[0,0])
    return np.array([phi, theta, psi])
 