"""robot_supervisor_manager controller."""

import SAC_runner
from gymnasium.envs.registration import register

if __name__ == '__main__':
    register(
        id="VektorE-v0",
        entry_point="robot_supervisor_controller:RobotSupervisorController"
    )
    SAC_runner.test_sac()
