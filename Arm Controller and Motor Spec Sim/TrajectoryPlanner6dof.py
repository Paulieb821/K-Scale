import mujoco as mj
import numpy as np
import math
import matplotlib.pyplot as plt
from NRIK import NRIK as ik

# --- Angle Key ---
# 0 0 90    --> points in +y
# 0 0 -90   --> points in -y
# 0 90 0    --> points in +x
# 0 -90 0   --> points in -x
# x 0 0 --> controls rotation about z?

class TrajectoryPlanner6dof: 

    def __init__(self, model, data, site, initialPose, pointsPerSecond):
        # Mujoco Stuff
        self.model = model
        self.data = data
        self.numJoints = model.nv
        # Site Location
        self.site = site
        # IK Solver
        self.ik = ik(model, data, self.site)
        # Integral Error Tracking
        self.accCounter = 0
        self.accCounterLimit = 100
        self.avgAcc = np.array([0, 0, 0, 0, 0, 0])
        self.F_guess = np.array([0, 0, 0, 0, 0, 0])[:, np.newaxis]
        # Stuff for Singularity Checking
        self.jacp = np.zeros((3, self.numJoints))
        self.jacr = np.zeros((3, self.numJoints))
        # Task Space Trajectory
        self.taskSpaceTraj = np.zeros([1, 7])
        self.taskSpaceVel = np.zeros([1,7])
        self.taskSpaceAcc = np.zeros([1,7])
        # Joint Space Trajectory
        self.jointSpaceTraj = np.zeros([1, self.numJoints])
        self.jointSpaceVel = np.zeros([1, self.numJoints])
        self.jointSpaceAcc = np.zeros([1, self.numJoints])
        # Controller References
        self.xref = -1
        self.uref = -1
        # Starting Point
        self.data.qpos = initialPose
        mj.mj_forward(self.model, self.data)
        # Load Starting Point into Arrays
        self.jointSpaceTraj[0,:] = np.array(initialPose)
        quatPlaceholder = np.zeros(4)
        mj.mju_mat2Quat(quatPlaceholder, self.site.xmat)
        self.taskSpaceTraj[0, :] = np.concatenate((self.site.xpos, quatPlaceholder))
        # Robot Performance Logging
        self.realTime = np.zeros(0)
        self.realPos = np.zeros([0, self.numJoints])
        self.estPos = np.zeros([0, self.numJoints]) 
        self.realVel = np.zeros([0, self.numJoints])
        self.estVel = np.zeros([0, self.numJoints])
        self.realAcc = np.zeros([0, self.numJoints])
        self.realTorque = np.zeros([0, self.numJoints])
        # Points per Second
        self.pps = pointsPerSecond
        # Keep Track of Path Time
        self.totalTime = 0
        self.timeArr = np.zeros(1)
        # Plotter
        self.plt = plt

    # Execute a Linear Point to Point Motion
    def addLinearMove(self, endPos, endRot, duration):
        # Number of Points in Trajectory
        numPoints = duration*self.pps+1
        
        # Convert Euler Angles to Quaterions to IK Solver
        startQuat = self.taskSpaceTraj[-1,:]
        endQuat = np.zeros(4)
        endRot = np.reshape(endRot, (9,1))
        mj.mju_mat2Quat(endQuat, endRot)
        endQuat = np.concatenate((endPos, endQuat))

        # Generate Quintic Task Space Trajectory
        localTaskSpaceTraj = np.zeros((numPoints, 7))
        localTaskSpaceVel = np.zeros((numPoints, 7))
        localTaskSpaceAcc = np.zeros((numPoints, 7))
        for coord in range(7):
            # Calculate Coefficients
            c3 = 10*(endQuat[coord] - startQuat[coord])/math.pow(duration,3)
            c4 = -15*(endQuat[coord] - startQuat[coord])/math.pow(duration,4)
            c5 = 6*(endQuat[coord] - startQuat[coord])/math.pow(duration,5)
            # Load Position into Array
            for i in range(numPoints):
                t = i/self.pps
                localTaskSpaceTraj[i,coord] = startQuat[coord] + c3*math.pow(t,3) + c4*math.pow(t,4) + c5*math.pow(t,5)
                localTaskSpaceVel[i, coord] = 3*c3*math.pow(t,2) + 4*c4*math.pow(t,3) + 5*c5*math.pow(t,4)
                localTaskSpaceAcc[i, coord] = 2*3*c3*math.pow(t,1) + 3*4*c4*math.pow(t,2) + 4*5*c5*math.pow(t,3)

        # Generate Joint Trajectories
        badTraj = False
        localJointSpaceTraj = np.zeros((numPoints, self.numJoints))
        localJointSpaceVel= np.zeros((numPoints, self.numJoints))
        localJointSpaceAcc = np.zeros((numPoints, self.numJoints))

        # Do Inverse Kinematics
        for i in range(numPoints):
            # Separate Position and Orientation
            pos = localTaskSpaceTraj[i, 0:3]
            ori = np.zeros(9)
            mj.mju_quat2Mat(ori, localTaskSpaceTraj[i, 3:7])
            # Calculate Inverse Position Kinematics and Load Into Array
            qpos = self.ik.solveIK(pos, ori)
            for joint in range(self.numJoints):
                localJointSpaceTraj[i, joint] = qpos[joint]
            # Update Starting Point for Good Convergence
            self.data.qpos = qpos
            mj.mj_forward(self.model, self.data)
        # Calculate Joint Velocities and Accelerations Numerically (maybe replace w/ Jacobian method?)
        for joint in range(self.numJoints):
            localJointSpaceVel[:,joint] = np.gradient(localJointSpaceTraj[:,joint], 1/self.pps)
            localJointSpaceAcc[:,joint] = np.gradient(localJointSpaceVel[:,joint], 1/self.pps)
        
        # Todo: Add a good way to detect singularities/bad trajectories!!

        # Add Motion to Master Trajectory
        self.taskSpaceTraj = np.concatenate((self.taskSpaceTraj, localTaskSpaceTraj[1:,:]))
        self.taskSpaceVel = np.concatenate((self.taskSpaceVel, localTaskSpaceVel[1:,:]))
        self.taskSpaceAcc = np.concatenate((self.taskSpaceAcc, localTaskSpaceAcc[1:,:]))

        self.jointSpaceTraj = np.concatenate((self.jointSpaceTraj, localJointSpaceTraj[1:,:]))
        self.jointSpaceVel = np.concatenate((self.jointSpaceVel, localJointSpaceVel[1:,:]))
        self.jointSpaceAcc = np.concatenate((self.jointSpaceAcc, localJointSpaceAcc[1:,:]))
        
        self.totalTime = self.totalTime + duration
        tempTimeArr = np.add(self.timeArr[-1], np.linspace(0, duration, numPoints))
        self.timeArr = np.concatenate((self.timeArr, tempTimeArr[1:]))

        self.generateReferences()

    # Hold a Position for a Set Amount of Time
    def addHold(self, duration):
        # Number of Points in Trajectory
        numPoints = duration*self.pps+1

        # Load Task Space Trajectories
        localTaskSpaceTraj = np.zeros((numPoints, 7))
        localTaskSpaceVel = np.zeros((numPoints, 7))
        localTaskSpaceAcc = np.zeros((numPoints, 7))
        for i in range(numPoints):
            for coord in range(7):
                localTaskSpaceTraj[i, coord] = self.taskSpaceTraj[-1, coord]

        # Generate Joint Trajectories
        localJointSpaceTraj = np.zeros((numPoints, self.numJoints))
        localJointSpaceVel= np.zeros((numPoints, self.numJoints))
        localJointSpaceAcc = np.zeros((numPoints, self.numJoints))
        # Load Joint Positions
        for i in range(numPoints):
            for joint in range(self.numJoints):
                localJointSpaceTraj[i, joint] = self.jointSpaceTraj[-1, joint]

        # Add Motion to Master Trajectory
        self.taskSpaceTraj = np.concatenate((self.taskSpaceTraj, localTaskSpaceTraj[1:,:]))
        self.taskSpaceVel = np.concatenate((self.taskSpaceVel, localTaskSpaceVel[1:,:]))
        self.taskSpaceAcc = np.concatenate((self.taskSpaceAcc, localTaskSpaceAcc[1:,:]))

        self.jointSpaceTraj = np.concatenate((self.jointSpaceTraj, localJointSpaceTraj[1:,:]))
        self.jointSpaceVel = np.concatenate((self.jointSpaceVel, localJointSpaceVel[1:,:]))
        self.jointSpaceAcc = np.concatenate((self.jointSpaceAcc, localJointSpaceAcc[1:,:]))
        
        self.totalTime = self.totalTime + duration
        tempTimeArr = np.add(self.timeArr[-1], np.linspace(0, duration, numPoints))
        self.timeArr = np.concatenate((self.timeArr, tempTimeArr[1:]))

        self.generateReferences()

    # Generate Discrete Time Controller References Before Running Controller
    def generateReferences(self):
        numPoints = np.size(self.timeArr)
        xref = np.zeros((numPoints, 2*self.numJoints))
        uref = np.zeros((numPoints, self.numJoints))
        # Populate first half of xref w/ position trajectory
        xref[:,0:self.numJoints] = self.jointSpaceTraj
        # Useful constants
        a = -2*self.pps
        b = 2*pow(self.pps,2)
        T = 1/self.pps
        # Solve for velocity and force references
        for j in range(self.numJoints):
            for i in range(numPoints-1):
                uref[i,j] = a*xref[i,self.numJoints+j] + b*(xref[i+1,j] - xref[i,j])
                xref[i+1,self.numJoints+j] = xref[i,self.numJoints+j] + T*uref[i,j]
        # Add to Trajectory 
        self.xref = xref
        self.uref = uref

    # Plot Joint Trajectories
    def plotJoints(self, motorSpecs = None):
        rpm_to_rads = 0.1047
        fig, axs = self.plt.subplots(4, self.numJoints)
        # Joints
        for row in range(4):
            for joint in range(self.numJoints):
                ax = axs[row, joint]
                if row == 0:
                    ax.plot(self.timeArr, self.xref[:,joint])
                    ax.plot(self.realTime, self.realPos[:, joint])
                    ax.plot(self.realTime, self.estPos[:, joint])
                elif row == 1:
                    ax.plot(self.timeArr, self.xref[:,self.numJoints+joint])
                    ax.plot(self.realTime, self.realVel[:, joint])
                    ax.plot(self.realTime, self.estVel[:, joint])
                elif row == 2:
                    ax.plot(self.realTime, self.realTorque[:,joint])
                else:
                    ax.plot(abs(self.realVel[:,joint]), abs(self.realTorque[:,joint]))
                    if motorSpecs != None:
                        ax.plot([0, motorSpecs[joint]["contw"]*rpm_to_rads], [motorSpecs[joint]["contT"], motorSpecs[joint]["contT"]], color='g')
                        ax.plot([motorSpecs[joint]["contw"]*rpm_to_rads, motorSpecs[joint]["contw"]*rpm_to_rads], [0, motorSpecs[joint]["contT"]], color='g')
                        ax.plot([0, motorSpecs[joint]["contw"]*rpm_to_rads], [motorSpecs[joint]["peakT"], motorSpecs[joint]["contT"]], color='r')
                        ax.plot([motorSpecs[joint]["peakw"]*rpm_to_rads, motorSpecs[joint]["contw"]*rpm_to_rads], [0, motorSpecs[joint]["contT"]], color='r')
        # Plot
        plt.show()





        