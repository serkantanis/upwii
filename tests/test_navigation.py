from __future__ import annotations

import unittest

from imu_nav.navigation import InertialNavigationEKF, NavConfig
from imu_nav.protocol import FrameHeader, FrameType, ImuSampleFrame, StatusFlag


VALID_STATUS = (
    StatusFlag.ACCEL_OK
    | StatusFlag.GYRO_OK
    | StatusFlag.IMU_CALIBRATED
    | StatusFlag.AHRS_OK
    | StatusFlag.LINEAR_ACCEL_OK
)


def frame(
    sequence: int,
    acceleration: tuple[float, float, float],
    *,
    accel_corrected_g: tuple[float, float, float] = (0.0, 0.0, 1.0),
    gyro_dps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ImuSampleFrame:
    return ImuSampleFrame(
        header=FrameHeader(1, FrameType.IMU_SAMPLE, 132, sequence, sequence * 5000),
        status=VALID_STATUS,
        delta_time_s=0.005,
        accel_raw_g=(0.0, 0.0, 1.0),
        accel_corrected_g=accel_corrected_g,
        gyro_raw_dps=gyro_dps,
        gyro_corrected_dps=gyro_dps,
        gyro_bias_applied_dps=(0.0, 0.0, 0.0),
        mag_gauss=(0.2, 0.0, 0.4),
        temperature_c=30.0,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        euler_deg=(0.0, 0.0, 0.0),
        linear_accel_world_mps2=acceleration,
        calibration_progress=1.0,
        mag_calibration_progress=1.0,
    )


class NavigationTests(unittest.TestCase):
    def test_one_second_constant_acceleration(self) -> None:
        config = NavConfig(stationary_hold_s=10.0, velocity_leak_per_s=0.0)
        navigator = InertialNavigationEKF(config)

        solution = None
        for sequence in range(200):
            solution = navigator.update(frame(sequence, (1.0, 0.0, 0.0)))

        assert solution is not None
        self.assertAlmostEqual(solution.velocity_mps[0], 1.0, places=2)
        self.assertAlmostEqual(solution.position_m[0], 0.5, places=2)

    def test_stationary_zupt_limits_drift(self) -> None:
        config = NavConfig(
            stationary_hold_s=0.10,
            velocity_leak_per_s=0.0,
            stationary_accel_threshold_mps2=0.30,
        )
        navigator = InertialNavigationEKF(config)

        solution = None
        for sequence in range(800):
            solution = navigator.update(frame(sequence, (0.05, -0.03, 0.02)))

        assert solution is not None
        self.assertTrue(solution.stationary)
        self.assertLess(solution.speed_mps, 0.01)
        self.assertLess(solution.distance_m, 0.01)
        self.assertGreater(navigator.zupt_updates, 100)

    def test_stationary_detection_ignores_gravity_removal_tilt_error(self) -> None:
        """AHRS tilt error may report linear acceleration while the unit rests."""
        config = NavConfig(
            stationary_hold_s=0.10,
            velocity_leak_per_s=0.0,
            stationary_accel_threshold_mps2=0.30,
        )
        navigator = InertialNavigationEKF(config)

        solution = None
        for sequence in range(400):
            solution = navigator.update(
                frame(sequence, (1.5, 0.0, 0.0), accel_corrected_g=(0.0, 0.0, 1.0))
            )

        assert solution is not None
        self.assertTrue(solution.stationary)
        self.assertLess(solution.speed_mps, 0.01)
        self.assertLess(solution.distance_m, 0.01)
        self.assertLess(abs(solution.position_m[0]), 0.01)


if __name__ == "__main__":
    unittest.main()
