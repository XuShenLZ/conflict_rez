# This is a message to hold data with system state information
std_msgs/Header header

int16                                  vehicle_id    # Vehicle ID

float64                                t       # Time

confrez_ros/PositionMsg                x

confrez_ros/OrientationEulerMsg        e # Orientation in Euler angles
confrez_ros/OrientationQuaternionMsg   q # Orientation in quaternions
confrez_ros/BodyAngularVelocityMsg     w # Vector of angular velocities
confrez_ros/BodyAngularAccelerationMsg aa # Vector of angular accelerations
confrez_ros/BodyLinearVelocityMsg      v # local body velocities
confrez_ros/BodyLinearAccelerationMsg  a # local body accelerations

confrez_ros/ParametricPoseMsg          p
confrez_ros/ParametricVelocityMsg      pt

confrez_ros/VehicleActuationMsg        u
