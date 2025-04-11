import mujoco
import numpy as np
import matplotlib.pyplot as plt

# --- Load your MJCF model ---
model = mujoco.MjModel.from_xml_path(r"C:\Users\aksha\Downloads\Github\K-Scale\Arm Controller and Motor Spec Sim\Sim_Arm_4DOF_Mar_25\robot.xml")
data = mujoco.MjData(model)

# --- Site name of the end-effector ---
site_name = "endeff"  # Replace with your actual site name

# --- Find site ID using mujoco API ---
# Get all site names from the model
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
print("This is the site id : ", site_id)
if site_id == -1:
    raise ValueError(f"Site '{site_name}' not found in the model")

# --- Define joint limits from model ---
joint_ranges = []
for j in range(model.nq):
    joint_ranges.append((model.jnt_range[j][0], model.jnt_range[j][1]))

# --- Sampling configurations ---
n_samples = 100000
positions = []

for _ in range(n_samples):
    # Sample random joint angles within joint limits
    qpos = np.array([np.random.uniform(low, high) for (low, high) in joint_ranges])
    data.qpos[:] = qpos

    # Forward kinematics to update site position
    mujoco.mj_forward(model, data)

    # Get end-effector site position using the site ID
    pos = data.site_xpos[site_id]
    positions.append(pos.copy())

positions = np.array(positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, alpha=0.4)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("4-DOF Robot Arm Workspace")
plt.show()