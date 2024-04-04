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
from geometry_msgs.msg import PointStamped

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
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/base_link_pf", 1) ## should be /base_link on car

        self.particles_pub = self.create_publisher(PoseArray, "/particles_vis", 1) # allows us to visualize particles

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
        # self.num_particles = self.get_parameter("num_particles").get_parameter_value().string_value
        self.num_particles = 200

        self.clicked_point = (0, 0)
        # set self.clicked_point to the point clicked in rviz
        # self.clicked_point = (x, y
        self.click_sub = self.create_subscription(PointStamped, '/clicked_point', self.click_callback, 10)
                    
        self.particles = np.zeros((self.num_particles, 3)) # todo this with the clicking

        self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().string_value
    
        # self._lock = threading.Lock()

    # msg_click of type PointStamped
    def click_callback(self, msg_click):
        x = msg_click.point.x
        y = msg_click.point.y
        self.get_logger().info("Point clicked at %d %d" % (x, y))

        self.particles[:, 0] = np.random.normal(x, 1.0, (self.num_particles,))
        self.particles[:, 1] = np.random.normal(y, 1.0, (self.num_particles,))
        self.particles[:, 2] = np.random.uniform(0, 2*np.pi, (self.num_particles,))

        self.get_logger().info("Created new Gaussian particles")
        self.update_particles_viz()
    
    def update_particles_viz(self):
        posearray = PoseArray()
        posearray.header.frame_id = "/map"
        for particle in self.particles:
            pose = Pose()

            # set position
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0.0

            quat = self.euler_to_quaternion(0,0,particle[2])

            # set rotation
            pose.orientation.w = quat[0]
            pose.orientation.x = quat[1]
            pose.orientation.y = quat[2]
            pose.orientation.z = quat[3] # unsure if this works tbh

            posearray.poses.append(pose)

        self.particles_pub.publish(posearray)

        self.get_logger().info("Published particles viz")

    # returns quaternion for a rotation given in euler angles
    # i got this online
    def euler_to_quaternion(self,roll, pitch, yaw):

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

# takes in param LaserScan
    def laser_callback(self, msg_laser):
        # https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

        return
         
        # self._lock.acquire()
        self.laserScan = msg_laser

        # update and correct 
        ranges = np.array(self.laserScan.ranges)
        self.num_beams_per_particle = 100
        laser_linspaced = ranges[np.linspace(0, ranges.size-1, int(self.num_beams_per_particle), dtype = int)]
        
        probabilities = self.sensor_model.evaluate(self.particles, laser_linspaced) 

        # resample particles from previous particles using probabilities (choice only takes in 1D array)
        indicies_particles = np.linspace(0, self.particles.shape[0]-1, self.particles.shape[0])
        # self.particles = np.random.choice(self.particles, self.num_particles, p=probabilities, replace=False)
        chosen_indicies = np.random.choice(indicies_particles, self.num_particles, p=probabilities, replace=False)
        for i in chosen_indicies:
            np.append(self.particles, self.particles[int(i),:])
        # could change to a different resampling method, but use this for now

        self.updatePoseGuess()

        # self._lock.release()

    # takes in param Odometry
    def odom_callback(self, odom):
        # https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html
      
        return
        # self._lock.acquire()

        if (self.laserScan == None): return # break out of function if no laser scans received yet

        twist = odom.twist.twist
        twist_x = twist.linear.x
        twist_y = twist.linear.y
        twist_theta = twist.angular.z
        
        self.particles = self.motion_model.evaluate(self.particles, np.array([twist_x, twist_y, twist_theta]))

        self.updatePoseGuess()

        # self._lock.release()

    def updatePoseGuess(self):
        # set pose to average of particles    
        
        # README HAS A COMMENT ABOUT A BETTER WAY TO FIND AVERAGES... especially if position has many modes in distribution
        self.pose = np.array([np.mean(self.particles[0]), np.mean(self.particles[1]),  np.arctan2(np.mean(np.sin(self.particles[2])), np.mean(np.cos(self.particles[2])))])
        print(self.pose)
        pose_toPub = PoseWithCovarianceStamped()

        pose_toPub.pose.pose.position.x = self.pose[0]
        pose_toPub.pose.pose.position.y = self.pose[1]
        pose_toPub.pose.pose.position.z = 0.0

        quat = self.euler_to_quaternion(0,0,self.pose[2])

        # set rotation
        pose_toPub.pose.pose.orientation.w = quat[0]
        pose_toPub.pose.pose.orientation.x = quat[1]
        pose_toPub.pose.pose.orientation.y = quat[2]
        pose_toPub.pose.pose.orientation.z = quat[3]

        
        self.pose_pub.publish(pose_toPub)

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
