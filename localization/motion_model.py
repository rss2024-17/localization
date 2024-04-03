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
        N = particles.size()
        noise_arr = np.random.normal(noise_mean, noise_stddev, (N, 3))

        for index, particle in enumerate(particles):
            # process particle

            #next position x, y (get by rotating dx, dy by particle theta and adding to particle x, y)
            theta = particle[2]
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])

            next_pos = particle[0:2] + rot_matrix * np.array([[odometry[0]], [odometry[1]]])
            
            #next theta (get it by adding particle theta and odometry theta)
            next_theta = particle[2]+odometry[2]
            
            next_particle = np.array([next_pos[0], next_pos[1], next_theta])

            particles[index] = next_particle

        particles += noise_arr

        ####################################
