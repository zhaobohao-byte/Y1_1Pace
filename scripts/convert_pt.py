import numpy as np
import torch
from pathlib import Path


def saveas_pt(npz_path, pt_path):
    """
    Convert sim_data.npz to chirp_data.pt format.

    Args:
        npz_path (str or Path): path to .npz file
        pt_path (str or Path): path to output .pt file
    """
    npz_path = Path(npz_path)
    pt_path = Path(pt_path)

    # 1. load npz
    data = np.load(npz_path)

    # 2. extract required fields
    time = torch.from_numpy(data["t"]).float()
    dof_pos = torch.from_numpy(data["q"]).float()
    des_dof_pos = torch.from_numpy(data["q_ref"]).float()

    # 3. save as pt
    torch.save(
        {
            "time": time,
            "dof_pos": dof_pos,
            "des_dof_pos": des_dof_pos,
        },
        pt_path
    )

    print(f"[OK] Saved pt file to: {pt_path}")

if __name__ == "__main__":
    # run the main function

    saveas_pt("/home/bohao/LuvRobot/Y1_1Pace/Y1_1Pace/scripts/pace/data/sim_data.npz", "/home/bohao/LuvRobot/Y1_1Pace/Y1_1Pace/scripts/pace/data/sim_data.pt")