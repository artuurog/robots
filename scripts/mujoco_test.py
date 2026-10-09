import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# File URDF nella stessa cartella dello script
urdf_path = Path("library\\ABB\\CRB15000_5kg_950_v1\\CRB15000_5kg_950.urdf").resolve()

# Carica il modello
model = mujoco.MjModel.from_xml_path(str(urdf_path))
data = mujoco.MjData(model)

# Giunti mobili del robot
joint_names = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

# Ricava gli indici delle coordinate qpos e qvel
joint_ids = []
for name in joint_names:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if joint_id == -1:
        raise ValueError(f"Giunto non trovato: {name}")
    joint_ids.append(joint_id)

qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])

# Posa desiderata in gradi: [joint_1, ..., joint_6]
q_target_deg = [0, -30, 45, 0, 45, 0]
q_target_rad = np.deg2rad(q_target_deg)

# Imposta la posa e azzera le velocità.
data.qpos[qpos_adr] = q_target_rad
data.qvel[dof_adr] = 0.0

# Aggiorna cinematica diretta, pose dei link e rendering.
mujoco.mj_forward(model, data)

print("Stato impostato:")
for name, q in zip(joint_names, data.qpos[qpos_adr]):
    print(f"{name}: {np.rad2deg(q):+.1f}° ({q:+.4f} rad)")

# Mostra il robot e mantieni aperta la finestra.
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()

    while viewer.is_running():
        time.sleep(0.01)