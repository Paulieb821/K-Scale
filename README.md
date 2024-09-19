# K-Scale
## Arm Controller and Motor Spec Sim
Follows an end effector trajector using a computed torque compensation and PD joint controller
Plots the following data about controller and motor performance:
- Joint position (planned, measured, and estimated)
- Joint velocity (planned, measured, and estimated)
- Joint torques
- Torque-speed curves (optionally with motor continous and peak limit)

## Tunable Parameters
Controller
- Discrete-time update frequency
- PD controller poles

Trajectory
- Design a trajectory composed on linear moves (quintic spline in time)
- Set end position as a vector and orientation as a rotation matrix (start is inherited from end of previous motion)
- Choose time to complete motion

Experimental/Misc
- Add a mass to the end of arm

![image](https://github.com/user-attachments/assets/640d42f3-9e9f-4ceb-8749-5c7e2789159c)

## Robot Models
Design Arms in Onshape and export as MJCF files using the [K-Scale Labs Onshape to MJCF/UDRF Converter](https://github.com/kscalelabs/onshape)
**Tips**
- Ensure arm is undirectional - one actuator is attached to the world frame and then each subsequent actuator is attached in series
- Make sure bodies aren't colliding with each other unintendedly - potentially exclude contact
- Add a site called "endeff" to the model so the IK knows what point to solve for
- Add visual coordinate frames to the base and end effector for easier debugging

![image](https://github.com/user-attachments/assets/d7761a10-0b98-45f4-94d9-92aca1bc4c17)

