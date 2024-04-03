import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        #################################### TODO

        # noise parameters; tweak
        noise_mean = 0
        noise_stddev = 0.5

        # generate noise array
        particles = np.array(particles)
        N = particles.shape[0]
        noise_arr = np.random.normal(noise_mean, noise_stddev, (N, 3))

        for index, particle in enumerate(particles):
            # process particle

            #next position x, y (get by rotating dx, dy by particle theta and adding to particle x, y)
            theta = particle[2]
            next_pos = particle[0:2] + np.array([np.cos(theta) * odometry[0] + -np.sin(theta) * odometry[1], 
                                                 np.sin(theta) * odometry[0] + np.cos(theta) * odometry[1]])

            #next theta (get it by adding particle theta and odometry theta)
            next_theta = (theta + odometry[2]) % (2 * 3.14159)

            next_particle = np.array([next_pos[0], next_pos[1], next_theta])

            particles[index] = np.round(next_particle, 2)

        # motion model tests only pass without added noise
        particles += np.round(noise_arr, 2)
       
        return (particles)

        ####################################
