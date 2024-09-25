import mujoco as mj
import numpy as np
import math
import os
from TrajectoryPlanner import TrajectoryPlanner6dof as traj

class Controller6dof:

    #def __init__(self, urdfAddress, initialPose, updateFreq, pole):
    def __init__(self, urdfAddress, site, initialPose, updateFreq, pole, calledAtRegInteval):
        # Set Up UDRF
        urdf_path = urdfAddress
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + urdf_path)
        urdf_path = abspath
        # Model and Data
        self.model = mj.MjModel.from_xml_path(urdf_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                 # MuJoCo data
        self.numJoints = self.model.nv
        self.site = self.data.site(site)
        # Make Update Frequency Available to Other Functions
        self.updateFreq = updateFreq
        self.T = 1/updateFreq
        # Does the controller get called by the sim/robot at a regular interval?
        self.calledAtRegInterval = calledAtRegInteval
        self.prevTime = 0
        # Most Recent Output Torque
        self.torqueOutput = np.zeros(self.numJoints)
        # Get Discrete Time State Equation Matrices and Gains
        self.getDTGains(pole)
        # Estimator State Variable
        self.xhat = np.array(initialPose)
        self.vhat = np.zeros(self.numJoints)
        # Dynamic Matrices
        self.M = np.zeros((self.numJoints, self.numJoints))
        self.updateDynamicMatrices()
        # Trajectory
        self.traj = self.createTrajectory()
        self.trajIndex = 0


    def createTrajectory(self):
        return traj(self.model, self.data, self.site, self.data.qpos, self.updateFreq)
    

    def trajectoryFollower(self, encoderVals, time = None):
        # Make Sure Function is Being Used Right
        if self.calledAtRegInterval == False and time == None:
            print("Please input time into function if loop doesn't update regularly")
            return np.zeros(self.numJoints)

        # Decide Whether to Update Torque Output Values
        if self.calledAtRegInterval == True or time == 0 or time - self.prevTime >= self.T:
            # Model Parameters
            self.updateDynamicMatrices()
            # Set Up Commanded Acceleration Vector
            cmdAcc = np.zeros((self.numJoints,1))
            # Run Joint-By-Joint DT PD Controller
            for j in range(self.numJoints):
                # Extract References
                if self.trajIndex <= self.traj.totalTime*self.traj.pps:
                    xref = self.traj.xref[self.trajIndex,j], 
                    vref = self.traj.xref[self.trajIndex,self.numJoints+j]
                    uref = self.traj.uref[self.trajIndex,j]
                else:
                    xref = self.traj.xref[self.trajIndex-1,j]
                    vref = 0
                    uref = 0
                # Command Acceleration
                cmdAcc[j] = uref - self.K1*(self.xhat[j] - xref) - self.K2*(self.vhat[j] - vref)
                # Update State Estimator - Replace Finite Difference!
                estError = self.xhat[j] - encoderVals[j]
                self.xhat[j] = self.xhat[j] + self.T*self.vhat[j] + 0.5*pow(self.T,2)*cmdAcc[j] - self.L1*estError
                self.vhat[j] = self.vhat[j] + self.T*cmdAcc[j] - self.L2*estError
            # Apply Compensation
            self.torqueOutput = (self.M @ cmdAcc + self.CG)[:, 0]
            #self.torqueOutput = self.traj.xref[self.trajIndex,:self.numJoints]
            # Update Previous Called Time
            if self.calledAtRegInterval == False:
                self.prevTime = time
            # Update Point Counter
            if self.trajIndex <= self.traj.totalTime*self.traj.pps:
                self.trajIndex = self.trajIndex + 1

        # Return Value
        return self.torqueOutput
        


    def getDTGains(self, pole):
        # Z-Transformed Pole
        gammaR = math.exp(-self.T*pole)
        gammaE = math.exp(-self.T*10*pole)
        # Regulator
        self.K1 = (2/self.T)*(1-gammaR)
        self.K2 = (2/pow(self.T,2))*(self.T*self.K1-1+pow(gammaR,2))
        # Estimator
        self.L1 = 2*(1-gammaE)
        self.L2 = (1/self.T)*(-1+self.L1+pow(gammaE,2)) 


    def updateDynamicMatrices(self):
        # Update Model State
        self.data.qpos = self.xhat
        self.data.qvel = self.vhat
        mj.mj_fwdPosition(self.model, self.data)
        mj.mj_fwdVelocity(self.model, self.data)

        # Get Inertia Matrix 
        mj.mj_fullM(self.model, self.M, self.data.qM)
        # Get Coriolis/Centripetal and Gravity Comp Matrix
        self.CG = self.data.qfrc_bias[:,np.newaxis]

