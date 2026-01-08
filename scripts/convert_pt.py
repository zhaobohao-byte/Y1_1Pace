import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def saveas_pt_with_noise(npz_path, pt_path, noise_std_ratio=0.05):
    """
    Convert sim_data.npz to chirp_data.pt format, adding tunable Gaussian white noise to dof_pos.
    Then plot the original and noisy data for visualization.

    Args:
        npz_path (str or Path): path to .npz file
        pt_path (str or Path): path to output .pt file
        noise_std_ratio (float): noise standard deviation ratio relative to the data's standard deviation.
                                 e.g., 0.05 means noise_std = 0.05 * std(dof_pos)
                                 Adjust this value to control noise amplitude (larger = more noise).
    """
    npz_path = Path(npz_path)
    pt_path = Path(pt_path)

    # 1. Load npz
    data = np.load(npz_path)
    
    # Extract fields (assuming keys: "t", "q", "q_ref")
    time_np = data["t"].astype(np.float32)
    dof_pos_np = data["q"].astype(np.float32)
    des_dof_pos_np = data["q_ref"].astype(np.float32)

    # 2. Add Gaussian white noise to measured position (dof_pos)
    data_std = np.std(dof_pos_np)
    actual_noise_std = noise_std_ratio * data_std
    print(f"Data std (dof_pos): {data_std:.6f}")
    print(f"Using noise std: {actual_noise_std:.6f} (ratio = {noise_std_ratio})")
    
    noise = np.random.normal(0, actual_noise_std, dof_pos_np.shape).astype(np.float32)
    dof_pos_noisy_np = dof_pos_np + noise

    # Convert to torch tensors
    time = torch.from_numpy(time_np)
    dof_pos_noisy = torch.from_numpy(dof_pos_noisy_np)
    des_dof_pos = torch.from_numpy(des_dof_pos_np)

    # 3. Save as .pt
    torch.save(
        {
            "time": time,
            "dof_pos": dof_pos_noisy,        # Noisy position saved
            "des_dof_pos": des_dof_pos,
        },
        pt_path
    )

    print(f"[OK] Saved noisy .pt file to: {pt_path}")

    # 4. Plot for visualization
    n_joints = dof_pos_np.shape[1]
    fig, axs = plt.subplots(n_joints, 1, figsize=(14, 4 * n_joints), sharex=True)
    if n_joints == 1:
        axs = [axs]

    # Handle time vector (flatten if necessary)
    time_plot = time_np.flatten() if time_np.ndim > 1 else time_np

    for i in range(n_joints):
        ax = axs[i]
        ax.plot(time_plot, des_dof_pos_np[:, i], label='Desired (q_ref)', linewidth=2, color='blue')
        ax.plot(time_plot, dof_pos_np[:, i], label='Original (q)', linewidth=1.5, color='green', alpha=0.8)
        ax.plot(time_plot, dof_pos_noisy_np[:, i], label=f'Noisy (q + noise, σ≈{actual_noise_std:.4f})', 
                linewidth=1, color='red', linestyle='--')
        ax.set_ylabel(f'Joint {i+1} Position (rad)')
        ax.legend(loc='upper right')
        ax.grid(True)

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Chirp Response with Added Gaussian Noise (noise_std_ratio = {noise_std_ratio})', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Adjustable noise level: change noise_std_ratio to control amplitude
    # 0.01 ~ very small noise, 0.1 ~ strong noise
    saveas_pt_with_noise(
        "/home/bohao/LuvRobot/Y1_1Pace/data/Y1_1_sim/chrip_data_mujoco.npz",
        "/home/bohao/LuvRobot/Y1_1Pace/data/Y1_1_sim/chrip_data_mujoco_noise.pt",
        noise_std_ratio=0.0002  # <-- Adjust this value as needed
    )