"""Minimal PID controller useful for testing the environment."""

from __future__ import annotations


class PIDController:
    """Discrete PID controller with integral clamping.

    Parameters
    ----------
    kp, ki, kd:
        Proportional, integral and derivative gains.
    dt:
        Control/simulation step in seconds.
    output_limits:
        Saturate the control signal to this range.
    integral_limits:
        Optional clamp on the integral term (anti-windup).
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.02,
        output_limits: tuple[float, float] = (-1.0, 1.0),
        integral_limits: tuple[float, float] | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error: float | None = None

    def __call__(self, error: float) -> float:
        self._integral += error * self.dt
        if self.integral_limits is not None:
            lo, hi = self.integral_limits
            self._integral = min(max(self._integral, lo), hi)

        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        lo, hi = self.output_limits
        return min(max(output, lo), hi)
