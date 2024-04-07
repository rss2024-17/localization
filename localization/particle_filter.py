from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy
import threading

assert rclpy

import numpy as np

# import laserscan message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.get_logger().info("INITIALIZING PARTICLE FILTER")
        
        # declare and retrieve our parameters
        self.declare_parameter('odom_topic', "/vesc/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('num_particles', "default")
        self.declare_parameter('num_beams_per_particle', "default")

        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().integer_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        # updates self.twist_x, self.twist_y, and self.twist_theta
        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)
        
        # function update_particles_odom actually updates the particles according to the odometry values
        # allows us to set the frequency at which we update odometry particles
        self.update_particles_odom_freq = 30.0 # in hertz
        self.create_timer(1/self.update_particles_odom_freq, self.update_particles_odom)
        self.twist_x = 0.0
        self.twist_y = 0.0
        self.twist_theta = 0.0

        self.update_particles_laser_freq = 20.0 # in hertz
        self.create_timer(1/self.update_particles_laser_freq, self.update_particles_laser)

        # print out all parameters and variables       
        self.get_logger().info("Reading laser scan from: %s" % scan_topic)
        self.get_logger().info("Reading odom data from: %s" % odom_topic)
        self.get_logger().info("Number of particles: %d" % self.num_particles)
        self.get_logger().info("Beams per particle: %d" % self.num_beams_per_particle)
        self.get_logger().info("Odom particle update freq: %d Hz" % self.update_particles_odom_freq)
        self.get_logger().info("Laser particle update freq: %d Hz" % self.update_particles_laser_freq)

        # Respond to pose initialization requests sent to the /initialpose topic
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  TODO *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.pose_pub = self.create_publisher(PoseStamped, "/base_link", 1) ## should be /base_link on car
        #self.laser_pose_pub = self.create_publisher(PoseStamped, "/laser", 1)

        # Publisher for us to visualize our particles
        self.particles_pub = self.create_publisher(PoseArray, "/particles_vis", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # TODO Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.pose = np.array([0,0,0]) # initialize pose as 0 (?)
        self.laserScan = None

        # listen for clicked points - initialize gaussian distribution of particles whenever clicked
        self.clicked_point = (0, 0)
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.click_callback, 10)
                    
        # before any clicks, just generate the particle distribution at the origin
        self.particles = np.zeros((self.num_particles, 3))
        self.generate_random_particles(0,0)
        self.update_particles_viz()

        self.get_logger().info("=============+READY+=============")

        # self._lock = threading.Lock()

    # msg_click of type PointStamped
    def click_callback(self, msg_click):
        x = msg_click.point.x
        y = msg_click.point.y
        self.get_logger().info("Point clicked at %f %f" % (x, y))
        self.generate_random_particles(x,y)
        self.update_particles_viz()
    
    def generate_random_particles(self, x, y):
        self.particles[:, 0] = np.random.normal(x, 1.0, (self.num_particles,))
        self.particles[:, 1] = np.random.normal(y, 1.0, (self.num_particles,))
        self.particles[:, 2] = np.random.uniform(0, 2*np.pi, (self.num_particles,))

        self.get_logger().info("Generated particles centered at %f %f" % (x,y))

    # publishes PoseArray visualization of current state of particles
    def update_particles_viz(self):
        if (self.particles_pub.get_subscription_count() == 0): return

        posearray = PoseArray()
        posearray.header.frame_id = "/map"
        for particle in self.particles:
            pose = Pose()

            # set position
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0.0

            quat = euler_to_quaternion(0,0,particle[2])

            # set rotation
            pose.orientation.w = quat[0]
            pose.orientation.x = quat[1]
            pose.orientation.y = quat[2]
            pose.orientation.z = quat[3]

            posearray.poses.append(pose)

        self.particles_pub.publish(posearray)

        # self.get_logger().info("Published particles visualization")

# takes in param LaserScan
    def laser_callback(self, msg_laser):
        # https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html
        self.laserScan = msg_laser
    
    def update_particles_laser(self):
        if (not self.laserScan): return
        if (not self.sensor_model.map_set): return

        # downsamples our laserscan (don't need all 1000+ range values)
        ranges = np.array(self.laserScan.ranges)
        laser_linspaced = ranges[np.linspace(0, ranges.size-1, int(self.num_beams_per_particle), dtype = int)]
        
        # retrieve probabilities corresponding to each particle
        probabilities = self.sensor_model.evaluate(self.particles, laser_linspaced)
        normalized_probabilities = probabilities / np.sum(np.array(probabilities))

        # resample particles from previous particles using probabilities (choice only takes in 1D array)
        indices_particles = np.linspace(0, self.particles.shape[0]-1, self.particles.shape[0])
        chosen_indices = np.random.choice(indices_particles, size=self.num_particles, p=normalized_probabilities, replace=True)

        # TODO do this with numpy
        new_particles = np.zeros((self.num_particles, 3))
        for i, chosen_index in enumerate(chosen_indices):
            new_particles[i] = self.particles[int(chosen_index)]
        self.particles = new_particles

        self.update_particles_viz()
        self.updatePoseGuess()

    # takes in param Odometry
    def odom_callback(self, odom):
        # https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html
        # self._lock.acquire()

        if (self.laserScan == None): return # break out of function if no laser scans received yet
        self.twist_x = -odom.twist.twist.linear.x
        self.twist_y = -odom.twist.twist.linear.y
        self.twist_theta = odom.twist.twist.angular.z

        # self._lock.release()

    def update_particles_odom(self):
        if (self.laserScan == None): return # break out of function if no laser scans received yet

        twist_x = self.twist_x / self.update_particles_odom_freq
        twist_y = self.twist_y / self.update_particles_odom_freq
        twist_theta = self.twist_theta / self.update_particles_odom_freq

        # self.get_logger().info("Odom: %f %f %f" % (twist_x, twist_y, twist_theta))
        self.particles = self.motion_model.evaluate(self.particles, np.array([twist_x, twist_y, twist_theta]))

        self.update_particles_viz()
        self.updatePoseGuess()

    def updatePoseGuess(self):
        # set pose to average of particles    
        # self.get_logger().info("Updating pose guess...")

        # README HAS A COMMENT ABOUT A BETTER WAY TO FIND AVERAGES... especially if position has many modes in distribution
        self.pose = np.array([np.mean(self.particles[0]), np.mean(self.particles[1]),  np.arctan2(np.mean(np.sin(self.particles[2])), np.mean(np.cos(self.particles[2])))])
        pose_toPub = PoseStamped()

        pose_toPub.header.frame_id = "/map"
        pose_toPub.pose.position.x = self.pose[0]
        pose_toPub.pose.position.y = self.pose[1]
        pose_toPub.pose.position.z = 0.0

        quat = euler_to_quaternion(0,0,self.pose[2])

        # set rotation
        pose_toPub.pose.orientation.w = quat[0]
        pose_toPub.pose.orientation.x = quat[1]
        pose_toPub.pose.orientation.y = quat[2]
        pose_toPub.pose.orientation.z = quat[3]

        self.pose_pub.publish(pose_toPub)
        #laser_pose_toPub = PoseStamped()
        #laser_pose_toPub.header.frame_id = "/map"
        #laser_pose_toPub.pose.position.x = 0.0
        #laser_pose_toPub.pose.position.y = 0.0
        #laser_pose_toPub.pose.position.z = 0.0
        #self.laser_pose_pub.publish(pose_toPub)

        # self.get_logger().info("Updated pose guess!")

    # takes in param PoseWithCovarianceStamped
    # override callback to set pose manually
    def pose_callback(self, update_pose):
        # https://docs.ros2.org/latest/api/geometry_msgs/msg/PoseWithCovarianceStamped.html
        update_pose_point = update_pose.pose.pose.position

        position = update_pose.pose.pose.position
        orientation = update_pose.pose.pose.orientation
        
        self.pose[0] = position.x
        self.pose[1] = position.y

        orientation = quaternion_to_euler(orientation.w, orientation.x, orientation.y, orientation.z)

        self.pose[2] = orientation[2] # we just want the yaw

        self.get_logger().info("Set pose to x: %f y: %f theta: %f" % (self.pose[0], self.pose[1], self.pose[2]))
    
# helper function to convert from euler angles to quaternion
def euler_to_quaternion(roll, pitch, yaw):

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w,x,y,z]

# helper function to convert from quaternion to euler angles
def quaternion_to_euler(w, x, y, z):

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.sqrt(1 + 2 * (w * y - x * z))
    cosp = np.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
