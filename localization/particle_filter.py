from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy

assert rclpy

import numpy as np

# import laserscan message
from sensor_msgs.msg import LaserScan

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #     *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.pose = np.array([0,0,0]) # initialize pose as 0 (?)
        self.laserScan = None

        # TODO: add a way to initialize these with a ui
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().string_value
        self.particles = np.random((self.num_particles, )) # todo this
        # but anyway particles is of shape (num_particles, 3)

        self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().string_value

# takes in param LaserScan
def laser_callback(self, msg_laser):
    # https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html
    self.laserScan = msg_laser

    # update and correct 
    ranges = np.array(self.laserScan.ranges)
    laser_linspaced = ranges[np.linspace(0, ranges.size()-1, num=self.num_beams_per_particle, dtype = int)]

    probabilities = self.sensor_model.evaluate(self.particles, laser_linspaced) 

    # resample particles from previous particles using probabilities
    self.particles = np.random.choice(self.particles, self.num_particles, p=probabilities, replaced=False)
    # could change to a different resampling method, but use this for now

    updatePoseGuess()

# takes in param Odometry
def odom_callback(self, odom):
    # https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html

    if (self.laserScan == None): return # break out of function if no laser scans received yet

    twist = odom.twist.twist
    twist_x = twist.linear.x
    twist_y = twist.linear.y
    twist_theta = twist.angular.z
    
    self.particles = self.motion_model.evaluate(self.particles, np.array([twist_x, twist_y, twist_theta]))

    updatePoseGuess(self)

def updatePoseGuess(self):
    # set pose to average of particles

    # TODO
    # for theta in self.particles[:, 2]:  https://en.wikipedia.org/wiki/Circular_mean
    #     if abs(theta) > np.pi/2:

    self.pose = np.array(np.mean(self.particles[0]), np.mean(self.particles[1]), np.mean(self.particles[2]))

# takes in param PoseWithCovarianceStamped
# override callback to set pose manually
def pose_callback(self, update_pose):
    # https://docs.ros2.org/latest/api/geometry_msgs/msg/PoseWithCovarianceStamped.html
    update_pose_point = update_pose.pose.pose.position
    self.pose[0] = update_pose_point.x
    self.pose[1] = update_pose_point.y
    self.pose[2] = update_pose_point.z

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
