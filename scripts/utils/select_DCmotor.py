import argparse
import matplotlib.pyplot as plt
import numpy as np


def calculate_curve(speed1, torque1, speed2, torque2):
    k = (torque2 - torque1) / (speed2 - speed1)
    saturation_torque = torque1 - k * speed1  # 虚线与y轴交点
    velocity_limit = -saturation_torque / k  # 虚线与x轴交点
    return saturation_torque, velocity_limit, k


def plot_curve(speed1, torque1, speed2, torque2, saturation_torque, velocity_limit, k):
    plt.figure(figsize=(10, 6))

    plt.plot([0, speed1], [torque1, torque1], "g-", linewidth=2)

    speeds = np.linspace(0, velocity_limit, 100)
    torques = saturation_torque + k * speeds
    plt.plot(speeds, torques, "r--", linewidth=2)

    plt.scatter(
        0,
        saturation_torque,
        s=80,
        zorder=5,
        c="red",
        label=f"saturation_torque ({saturation_torque:.2f})",
    )
    plt.scatter(
        speed1,
        torque1,
        s=80,
        zorder=5,
        c="green",
        label=f"TP ({speed1:.1f}, {torque1:.1f})",
    )
    plt.scatter(
        speed2,
        torque2,
        s=80,
        zorder=5,
        c="purple",
        label=f"CP ({speed2:.1f}, {torque2:.1f})",
    )
    plt.scatter(
        velocity_limit,
        0,
        s=80,
        zorder=5,
        c="orange",
        label=f"velocity_limit ({velocity_limit:.2f})",
    )

    plt.xlabel("Speed [rpm]")
    plt.ylabel("Torque [Nm]")
    plt.title("T-n Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="电机转矩-转速特性曲线计算")
    parser.add_argument(
        "--TP",
        nargs=2,
        type=float,
        default=[120, 9],
        metavar=("SPEED", "TORQUE"),
        help="恒转矩转折点 (默认: 120 10)",
    )
    parser.add_argument(
        "--CP",
        nargs=2,
        type=float,
        default=[90, 12],
        metavar=("SPEED", "TORQUE"),
        help="恒功率点 (默认: 7 3)",
    )
    args = parser.parse_args()

    speed1, torque1 = args.TP
    speed2, torque2 = args.CP

    saturation_torque, velocity_limit, k = calculate_curve(
        speed1, torque1, speed2, torque2
    )

    print(f"\nsaturation_torque: {saturation_torque:.2f} Nm")
    print(f"velocity_limit: {velocity_limit:.2f} rpm\n")

    plot_curve(speed1, torque1, speed2, torque2, saturation_torque, velocity_limit, k)