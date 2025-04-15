# Mujoco Imports
import mujoco as mj
from mujoco.glfw import glfw
# Generic Python Imports
import math
import numpy as np
import os
# My Class Imports
from Controller import Controller6dof as ctrl

 
###############
# SETUP STUFF #
###############

# Clear Terminal Before Run
os.system("cls")

# XML File and Simulation Time
#xml_path = 'robot/v2_asm.xml' #xml file (assumes this is in the same folder as this file)
#xml_path = '131-1x3-none/sim_arm_131-1x3-none.xml' #xml file (assumes this is in the same folder as this file)
# xml_path = '4dof_arm_v2/4dof_arm_v2.xml' #xml file (assumes this is in the same folder as this file)
xml_path = 'Sim_Arm_4DOF_Mar_25/robot.xml'
simend = 60 # simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# Get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)

# MuJoCo data structures
model = mj.MjModel.from_xml_path(abspath)  # MuJoCo model
data = mj.MjData(model)                     # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options
site_name = "endeff"  
site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
print("Before step:", data.site_xpos[site_id])  # likely [0, 0, 0]
mj.mj_step(model, data)
print("After step:", data.site_xpos[site_id])   # should now be real position

#############################
# Controller and Trajectory #
#############################   

# Declare Controller
# armController = ctrl(xml_path, "endeff", [-math.pi, -math.pi/3, -math.pi/3, 0.0, 0.0, 0.0], 100, 10, False)
armController = ctrl(xml_path, site="endeff", initialPose=[0.0, 0.0, 0.0, 0.0], updateFreq=1000, pole=10, calledAtRegInteval=False)

# Set Trajectory
# 4-DOF Bot Example
radius = 0.3
circle_start = np.array([0.0, 0.35, 0]) # 


armController.traj.addLinearMove_3dof(circle_start, 2)
armController.traj.trace_circle(radius, 5)


# armController.traj.addLinearMove_3dof(np.array([0, 0.2, 0.7]), 2)
# armController.traj.addLinearMove_3dof(np.array([0, 0.7, 0.2]), 2)
# armController.traj.addLinearMove_3dof(np.array([0, 0.4, 0.4]), 2)
# armController.traj.addLinearMove_3dof(np.array([0.6, 0.3, 0.1]), 2)
# armController.traj.addLinearMove_3dof(np.array([-0.6, 0.3, 0.1]), 2)
# armController.traj.addLinearMove_3dof(np.array([0, 0.4, 0.4]), 2)

# 6-DOF Bot Example
#armController.traj.addLinearMove_6dof(np.array([0, 0.4, -0.2]), np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) , 1)
#armController.traj.addLinearMove_6dof(np.array([0, 0.4, 0.4]), np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) , 1)
#armController.traj.addLinearMove_6dof(np.array([0.4, 0.4, 0.4]), np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) , 1)
#armController.traj.addLinearMove_6dof(np.array([0.4, 0.4, -0.2]), np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) , 1)

# Motor Specs
motorSpecs = [{"contT": 6, "contw" : 275, "peakT": 17, "peakw": 315, "Kt": 1.22, "R": 0.58},
            {"contT": 20, "contw" : 180, "peakT": 60, "peakw": 195, "Kt": 2.36, "R": 0.39},
            {"contT": 6, "contw" : 275, "peakT": 17, "peakw": 315, "Kt": 1.22, "R": 0.58},
            {"contT": 18, "contw" : 91, "peakT": 51, "peakw": 105, "Kt": 3.66, "R": 0.58}]

####################################
# Mujoco Controller Implementation #
####################################

def init_controller(model,data):
    # Set Arm Initial Point
    data.qpos = armController.xhat
    mj.mj_forward(model, data)



def controller(model, data):
        # Control Signal
        data.qfrc_applied = armController.trajectoryFollower(data.qpos, data.time)

        # Logging
        # if(data.time <= armController.traj.totalTime):
        armController.traj.realTime = np.append(armController.traj.realTime, data.time)
        armController.traj.realPos = np.concatenate((armController.traj.realPos, [data.qpos]), axis=0)
        armController.traj.estPos = np.concatenate((armController.traj.estPos, [armController.xhat]), axis=0)
        armController.traj.realVel = np.concatenate((armController.traj.realVel, [data.qvel]), axis=0)
        armController.traj.estVel = np.concatenate((armController.traj.estVel, [armController.vhat]), axis=0)
        armController.traj.realAcc = np.concatenate((armController.traj.realAcc, [data.qacc]), axis=0)
        armController.traj.realTorque = np.concatenate((armController.traj.realTorque, [data.qfrc_applied]), axis=0)
        armController.traj.endEffPos = np.concatenate((armController.traj.endEffPos, [data.site_xpos[site_id]]), axis = 0)
            
#####################   
# SIMULATION AND UI #
#####################

def keyboard(window, key, scancode, act, mods): 
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
#cam.azimuth = 180
#cam.elevation = 0
cam.azimuth = 135
cam.elevation = -15
cam.distance = 2
cam.lookat = np.array([0, 0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)

    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()

armController.traj.plotJoints(motorSpecs=motorSpecs)

# armController.traj.batteryCalc(36, 2, motorSpecs, 1)