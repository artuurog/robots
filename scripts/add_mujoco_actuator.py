import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

urdf_path = Path("library\\ABB\\CRB15000_5kg_950_v1\\CRB15000_5kg_950.urdf").resolve()

joint_names = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

# Limiti letti dal tuo URDF: [min, max] in rad
joint_ranges = [
    [-3.14159,  3.14159],
    [-3.14159,  3.14159],
    [-3.92699,  0.95993],
    [-3.14159,  3.14159],
    [-3.14159,  3.14159],
    [-4.71239,  4.71239],
]

# Coppie massime [Nm] dai limiti effort dell'URDF
max_torque = [175.44, 175.44, 90.6, 18.72, 21.44, 9.2]

# 1. Carica l'URDF come modello modificabile
spec = mujoco.MjSpec.from_file(str(urdf_path))

# 2. Aggiunge un attuatore di posizione a ogni joint
for joint_name, ctrl_range, torque_limit in zip(
    joint_names, joint_ranges, max_torque
):
    actuator = spec.add_actuator(
        name=f"{joint_name}_position",
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )

    # data.ctrl contiene il riferimento di posizione [rad]
    actuator.set_to_position(kp=100.0)

    # Limita comando e coppia prodotta dal servo
    actuator.ctrllimited = True
    actuator.ctrlrange = ctrl_range
    actuator.forcelimited = True
    actuator.forcerange = [-torque_limit, torque_limit]

# 3. Compila: ora model.nu == 6 e data.ctrl ha sei elementi
model = spec.compile()
data = mujoco.MjData(model)

print("Numero di attuatori:", model.nu)
for i, joint_name in enumerate(joint_names):
    print(f"data.ctrl[{i}] -> {joint_name}")

# Posa desiderata iniziale [deg]
q_target = np.deg2rad([0, -30, 45, 0, 45, 0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Comando dei sei attuatori di posizione, in radianti
        data.ctrl[:] = q_target

        # Esempio: muovi il primo giunto sinusoidalmente
        data.ctrl[0] = 0.5 * np.sin(0.5 * data.time)

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        time.sleep(max(0.0, model.opt.timestep - elapsed))

model = spec.compile()
output_path = urdf_path.with_name(
    "CRB15000_5kg_950_with_actuators.xml"
)
spec.encode(str(output_path))

print(f"Modello MJCF con attuatori salvato in: {output_path}")