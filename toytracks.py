import numpy as np
import matplotlib.pyplot as plt

"""
This contains the skeleton for the generation of any general track, provided its fucntion (or piecewise functions) can be parametrically defined. As long as that function can be defined, `generate_centroid' will create a central spline of points, and then `diffuse' will apply the appropriate gaussian diffusion, as defined by the covariance matrix.

This is currently set up to generate the high/low deposition U-shaped tracks, but varying the sequence of functions generated in the main `savetrack' function.
"""

covariance = np.array(([0.026**2,0,0],[0,0.026**2,0],[0,0,0.016**2]))

drift_vel = 0.013


global N1, N2, pathname

strip_width = (10/120)
# #electron params

#path to save outputs to
pathname = "/Users/magnus/Documents/MIGDAL/artificial_tracks/add_u_shaped_high_diff_inputs/u_high_diff_"

#electron practical range, and number of electrons
line_length = 0.52
N1 = 175


#F+ ion range and number of ions
line_length_2 = 0.568
N2 = 7289


def generate_centroid(function, start, stop, N,a,b,c,x0=0,y0=0,z0=0):
    """
    generates the centroid of a track based on a parametrically defined function
    parameters
    --------
    function: The function of the parametric variable, which returns coordinates in x, y, z
    start: start of parametric variable t
    stop: end of parametric variable t
    step: step interval
    returns
    --------
    a Nx3 linear numpy array of points
    """
    t = np.linspace(start, stop, N)
    output = np.zeros((N,3))

    print(a,b,c)
    for i in range(N):
        output[i] = function(t[i],x0=x0,y0=y0,z0=z0,a=a,b=b,c=c)

    return(output)

def diffuse(centroid_points,covariance,m):
    """
    Diffuses m points from each point of the centroid, assuming a gaussian spread with standard deviation sigma in each dimension
    """
    N = len(centroid_points)
    final_points = np.zeros((N*m,3))
    index = 0
    for i in range(N):
        drift_z = 1.5+centroid_points[i][2]
        for j in range(N2):
            final_points[index] = np.random.multivariate_normal(centroid_points[i],covariance*(drift_z**0.5))
            index += 1

    return(final_points)

def generate_line_params(theta,phi):
    """
    Assuming a line x=at+x0, y=bt+y0, z=ct+z0 etc. this generates a,b,c for polar theta, phi.
    """
    a = np.sin(theta)*np.cos(phi)
    b = np.sin(theta)*np.sin(phi)
    c = np.cos(theta)
    return(a,b,c)

def line(t,x0=0,y0=0,z0=0,a=1,b=1,c=1):
    x = x0 + a*t
    y = y0 + b*t
    z = z0 + c*t
    return([x,y,z])

def generate_circle(radius,N_points,inclination = 0):
    """
    Circle segments, with differing radii and inclination (rotation about the x-axis)
    """
    thetas = np.linspace(0,line_length/radius,N_points)-np.pi/2

    output = np.zeros((N_points,3))

    rotation = np.array([[1,0,0],
                [0,np.cos(inclination),-np.sin(inclination)],
                [0,np.sin(inclination),np.cos(inclination)]])
    
    for i in range(N_points):
        out = radius*np.array([np.cos(thetas[i]),np.sin(thetas[i]),0])
        out = np.dot(rotation,out)
        output[i] = out

    return(output)

    
    

def savetrack(theta,phi,x0=0,y0=0,z0=0,i=0,ratio=0):

    a1,b1,c1 = generate_line_params(theta,phi)

    a = a1
    b = (np.cos(i)*b1)-(np.sin(i)*c1)
    c = (np.sin(i)*b1)+(np.cos(i)*c1)

    centroid_points = generate_centroid(line, 0, line_length, int(N1),a,b,c,x0,y0,z0)
    centroid_points = np.concatenate((centroid_points,generate_centroid(line, 0, line_length_2,int(N2),a,b,c,x0,-y0,z0)))
    centroid_points = np.concatenate((centroid_points,generate_centroid(line, 0, 2*y0,int(N1*2*(y0/line_length)),b,-a,c,x0,y0,z0)))

    # centroid_points = generate_circle(line_length*ratio/(2*np.pi),N1,np.pi/2)
    # centroid_points = [[0,0,0]]
    raw_points = np.transpose(diffuse(centroid_points,np.zeros((3,3)),1))
    final_points = np.transpose(diffuse(centroid_points,covariance,1))

    diffused_time = final_points[2]/drift_vel
    diffused_time = diffused_time - np.min(diffused_time)
    
    output = np.zeros((6,len(centroid_points)))
    output[:3] = raw_points
    output[3:5] = final_points[0:2]
    output[5] = diffused_time


    filename = pathname+"sep_"+str("%.3f" % (2*y0)).replace('.', '_')+".txt"
    np.savetxt(filename,np.transpose(output),fmt='%.4f',delimiter=" ")

     fig = plt.figure()
     ax2 = plt.axes(projection='3d')

     scatter_plot = ax2.scatter3D(final_points[0],final_points[1],final_points[2],label="Diffused")
     scatter_plot2 = ax2.scatter3D(raw_points[0],raw_points[1],raw_points[2],label="Undiffused")

     plt.legend()

     ax2.set_xlabel("x [cm]")
     ax2.set_ylabel("y [cm]")
     ax2.set_zlabel("z [cm]")

     plt.show()


thetas = np.linspace(np.pi/7,np.pi*(6/7),6)
phis = np.linspace(0,np.pi,7)

#14 U-shaped tracks with lines between 0.125 and 0.175cm in separation.
separations = np.linspace(0.125,0.175,14)

for separation in separations:
    savetrack(np.pi/2,0, x0=-2*strip_width,y0=separation/2)
