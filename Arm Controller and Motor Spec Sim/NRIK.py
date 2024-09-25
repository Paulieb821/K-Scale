import mujoco as mj
import math
import numpy as np
from scipy.linalg import logm

class NRIK:

    def __init__(self, model, data, site):
        # Mujoco
        self.model = model
        self.data = data
        self.site = site
        self.numJoints = model.nv
        # Jacobian Setup
        self.jacp = np.zeros((3,self.numJoints))
        self.jacr = np.zeros((3,self.numJoints))

    # Inverse Kinematics for Position and Pose
    def solveIK_6dof(self, endPos, endRot):
        # Set error bounds and iteration limit
        posErrLim = 0.001
        rotErrLim = 0.001
        limit = 100
        # Initialization
        q = self.data.qpos
        i = 0
        posErr = 100
        rotErr = 100
        # Algorithm
        while((posErr > posErrLim or rotErr > rotErrLim) and i <= limit):
            # Get Body Twist from Current Position to Desired Position
            currentPoseSpace = self.toHTM(self.site.xpos, self.site.xmat)
            endPoseSpace = self.toHTM(endPos, endRot)
            endPoseBody = np.matmul(np.linalg.inv(currentPoseSpace), endPoseSpace)
            VbSS = logm(endPoseBody)
            Vb = np.zeros((6,1))
            Vb[0] = VbSS[2,1]
            Vb[1] = VbSS[0,2]
            Vb[2] = VbSS[1,0]
            Vb[3] = VbSS[0,3]
            Vb[4] = VbSS[1,3]
            Vb[5] = VbSS[2,3]
            # Get Body Jacobian
            jacBody = self.getBodyJacobian()
            # Do Newton-Rhapson
            q = np.add(q, np.transpose(np.matmul(np.linalg.pinv(jacBody), Vb)))
            self.data.qpos = q
            mj.mj_forward(self.model, self.data)
            #Compute Error
            rotErr = np.linalg.norm(Vb[:3])
            posErr = np.linalg.norm(Vb[3:])
            # Increment Counter
            i += 1
        # Return Conditions
        if i > limit:
            print("Failed to Converge")
            return self.data.qpos
        else:
            return self.data.qpos
        
    # Inverse Kinematics for Position Only
    def solveIK_3dof(self, endPos):
        # Set error bounds and iteration limit
        posErrLim = 0.001
        limit = 100
        # Initialization
        q = self.data.qpos
        i = 0
        posErr = 100
        # Algorithm
        while(posErr > posErrLim and i <= limit):
            # Get Relative Vector Between Current and End Pose
            relPos = endPos - self.site.xpos
            # Get Space Jacobian
            mj.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site.id)
            # Do Newton Rhapson
            q = np.add(q, np.matmul(np.linalg.pinv(self.jacp), relPos))
            self.data.qpos = q
            mj.mj_forward(self.model, self.data)
            # Check error
            posErr = np.linalg.norm(endPos - self.site.xpos)
            # Increment Counter
            i += 1
        # Return Conditions
        if i > limit:
            print("Failed to Converge")
            return self.data.qpos
        else:
            return self.data.qpos
            
    def getBodyJacobian(self):
        # Get Space Jacobian
        mj.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site.id)
        jacSpace = np.concatenate((self.jacr, self.jacp))
        # Get Adjoint Matrix
        adjoint = self.adjointMatrix()
        # Multiply Adjoint Matrix by Space Jacobian
        jacBody = np.matmul(adjoint, jacSpace)
        return jacBody

    def toHTM(self, xpos, xmat):
        htm = np.zeros((4,4))
        htm[3,3] = 1
        # Load Rotation Matrix
        rotMat = np.resize(xmat, (3,3))
        for i in range(3):
            for j in range(3):
                htm[i,j] = rotMat[i,j]
        # Load Position Vector
        for i in range(3):
            htm[i,3] = xpos[i]
        # Output
        return htm
    
    def adjointMatrix(self):
        # Get Skew Symmetric Matrix form of Position
        p = self.site.xpos
        pSS = [[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]]
        # Get Rotation Matrix
        rotMat = self.site.xmat
        rotMat = np.reshape(rotMat, (3, 3))
        # Declare Adjoint Matrix
        adjoint = np.zeros((6,6))
        # Load Rotation Matrix
        for i in range(3):
            for j in range(3):
                adjoint[i,j] = rotMat[i,j]
                adjoint[i+3,j+3] = rotMat[i,j]
        # Load [p]R Matrix
        pR = np.matmul(pSS, rotMat)
        for i in range(3):
            for j in range(3):
                adjoint[i+3,j] = pR[i,j]
        # Return Matrix
        return adjoint
