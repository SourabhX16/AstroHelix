# src/physics/solver.py
import numpy as np
from engine.physics.state import State

class RK4Solver:
    """
    Runge-Kutta 4th order solver for state integration.
    """
    def __init__(self, force_model):
        self.force = force_model

    def step(self, state, dt):
        """
        Perform a single RK4 step.
        """
        position = state.r
        velocity = state.v

        def acceleration_at(stage_position, stage_velocity):
            stage = State.__new__(State)
            stage.r = stage_position
            stage.v = stage_velocity
            return self.force.acceleration(stage)

        k1_position = velocity
        k1_velocity = self.force.acceleration(state)
        k2_position = velocity + 0.5 * dt * k1_velocity
        k2_velocity = acceleration_at(position + 0.5 * dt * k1_position, k2_position)
        k3_position = velocity + 0.5 * dt * k2_velocity
        k3_velocity = acceleration_at(position + 0.5 * dt * k2_position, k3_position)
        k4_position = velocity + dt * k3_velocity
        k4_velocity = acceleration_at(position + dt * k3_position, k4_position)

        next_position = position + (dt / 6.0) * (
            k1_position + 2 * k2_position + 2 * k3_position + k4_position
        )
        next_velocity = velocity + (dt / 6.0) * (
            k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity
        )
        return State(next_position, next_velocity)
