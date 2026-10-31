"""Run the 1D environment with a hand-tuned PID controller."""

from __future__ import annotations

import argparse

from pid_env import EnvironmentConfig, PID1DEnv, PIDController, plot_trajectory


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kp", type=float, default=3.0)
    p.add_argument("--ki", type=float, default=0.8)
    p.add_argument("--kd", type=float, default=1.2)
    p.add_argument("--target", type=float, default=1.0)
    p.add_argument("--disturbance", type=float, default=4.0, help="noise force strength (N)")
    p.add_argument("--target-noise", type=float, default=0.0, help="moving setpoint amplitude (m)")
    p.add_argument("--thrust", type=float, default=20.0)
    p.add_argument("--drag", type=float, default=0.5)
    p.add_argument("--weight", type=float, default=9.81)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", type=str, default=None, help="save a plot to this path")
    p.add_argument("--show", action="store_true", help="open an interactive plot window")
    return p


def main() -> None:
    args = build_parser().parse_args()

    config = EnvironmentConfig(
        weight=args.weight,
        drag=args.drag,
        thrust=args.thrust,
        target=args.target,
        disturbance_strength=args.disturbance,
        disturbance_frequency=1.5,
        target_noise_strength=args.target_noise,
        target_noise_frequency=0.3,
        bounds=(-5.0, 5.0),
        reward_control_cost=0.01,
    )

    env = PID1DEnv(config)
    env.reset(seed=args.seed)
    pid = PIDController(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        dt=config.dt,
        integral_limits=(-2.0, 2.0),
    )

    times: list[float] = []
    positions: list[float] = []
    targets: list[float] = []
    velocities: list[float] = []
    actions: list[float] = []

    total_reward = 0.0
    while True:
        action = pid(env.error)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        times.append(info["time"])
        positions.append(info["position"])
        targets.append(info["target"])
        velocities.append(info["velocity"])
        actions.append(action)

        if terminated or truncated:
            break

    abs_error = sum(abs(p - t) for p, t in zip(positions, targets)) / len(positions)
    print(f"steps         : {len(positions)}")
    print(f"final position: {positions[-1]:.4f}")
    print(f"final target  : {targets[-1]:.4f}")
    print(f"mean |error|  : {abs_error:.4f}")
    print(f"total reward  : {total_reward:.2f}")

    if args.plot or args.show:
        plot_trajectory(
            positions,
            targets,
            times=times,
            velocities=velocities,
            actions=actions,
            title="1D PID simulation",
            save_path=args.plot,
            show=args.show,
        )
        if args.plot:
            print(f"saved plot to : {args.plot}")


if __name__ == "__main__":
    main()
