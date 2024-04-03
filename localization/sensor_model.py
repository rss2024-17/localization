import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0
        self.alpha_short = 0
        self.alpha_max = 0
        self.alpha_rand = 0
        self.sigma_hit = 0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        self.z_max = 10

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model(node)

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self, node):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        z_max = 10
        self.table_width=201

        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.table_width = 201

        #Each (i, j) corresponds to the sensor model probability 
        # of measuring z=(j*z_max/200) given d=(i*z_max/200)
        self.axis = np.linspace(0, z_max, self.table_width)
        # self.z = np.repeat(np.linspace(0, z_max, self.table_width).reshape(1, -1), self.table_width, axis=0)
        # self.d = np.repeat(np.linspace(0, z_max, self.table_width).reshape(-1, 1), self.table_width, axis=1)
        
        # create max table
        self.max_table = np.zeros((self.table_width, self.table_width))
        for i in range(0, self.table_width):
            self.max_table[i, self.table_width-1] = 1
        # self.max_table[:, -1]=1
        node.get_logger().info(f"{self.max_table[0:5,0:5]}")
        
        # create and normalize hit table
        self.hit_table = np.zeros((self.table_width, self.table_width))
        for i in range(0, self.table_width):
            for j in range(0, self.table_width):
                d = self.axis[i]
                z = self.axis[j]
                self.hit_table[i][j] = np.exp(-1*((z-d)**2)/(2*self.sigma_hit**2))
        # for j in range(0, self.table_width): # for each col
        #     sum = 0
        #     for i in range(0, self.table_width):
        #         sum += self.hit_table[i][j]
        #     for i in range(0, self.table_width):
        #         self.hit_table[i][j] /= sum

        # for j in range(0, self.table_width):
        #     norm = np.sum(self.hit_table[:, j])
        #     for i in range(0, self.table_width):
        #         self.hit_table[i][j] /= norm
        for i in range(0, self.table_width):
            norm = np.sum(self.hit_table[i, :])
            for j in range(0, self.table_width):
                self.hit_table[i][j] /= norm

        # sums = np.sum(self.hit_table, axis=0)
        # sums = np.tile(sums, (self.table_width, 1))
        # self.hit_table  = self.hit_table / sums
        # normalizing_constants = 1/np.repeat(self.hit_table.sum(axis=-1).reshape(-1, 1), self.table_width, axis=1) # check this
        # self.hit_table = self.hit_table*normalizing_constants
        node.get_logger().info(f"{self.hit_table[0:5,0:5]}")
        
        # create rand table
        self.rand_table = np.zeros((self.table_width, self.table_width))
        # self.rand_table = self.rand_table + 1/self.table_width
        self.rand_table = self.rand_table + 1/self.z_max
        node.get_logger().info(f"{self.rand_table[0:5,0:5]}")
        
        # create and normalize short table
        self.short_table = np.zeros((self.table_width, self.table_width))
        for i in range(0, self.table_width):
            for j in range(0, self.table_width):
                d = self.axis[i]
                z = self.axis[j]
                if (z <= d and not d == 0):
                    self.short_table[i][j] = 2/d * (1-z/d)
        # normalizing_constants = 1/np.repeat(self.short_table.sum(axis=-1).reshape(-1, 1), self.table_width, axis=1)
        # self.short_table = self.short_table*normalizing_constants
        node.get_logger().info(f"{self.short_table[0:5,0:5]}")

        self.sensor_model_table = self.alpha_hit*self.hit_table + self.alpha_short*self.short_table + self.alpha_max*self.max_table + self.alpha_rand*self.rand_table
        # self.sensor_model_table = 1/np.repeat(self.sensor_model_table.sum(axis=-1).reshape(-1, 1), self.table_width, axis=1)
        # sums = np.sum(self.sensor_model_table, axis=0)
        # sums = np.tile(sums, (self.table_width, 1))
        # self.sensor_model_table  = self.sensor_model_table / sums
        for j in range(0, self.table_width):
            norm = np.sum(self.sensor_model_table[:, j])
            for i in range(0, self.table_width):
                self.sensor_model_table[i][j] /= norm

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles) # computes the simulated laserScan for each particle

        new_observation = []
        for obs in observation:
            if obs > self.z_max:
                obs = self.z_max
            elif obs < 0:
                obs = 0
            o = obs/(self.map_metadata.resolution * self.lidar_scale_to_map_scale)
            new_observation.append(o)

        probabilities = []
        for scan in scans:
            probability = 1
            for index, point in enumerate(scan):
                probability *= self.sensor_model_table[point, new_observation[index]]
            probabilities.append(probability)
        
        return probabilities

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)
        self.map_metadata = map_msg.info

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
