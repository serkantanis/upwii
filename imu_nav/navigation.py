"""Short-term inertial dead reckoning with ZUPT and a 9-state EKF."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .protocol import ImuSampleFrame, StatusFlag


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class NavConfig:
    acceleration_noise_mps2: float = 0.35
    accel_bias_random_walk_mps3: float = 0.015
    zupt_velocity_sigma_mps: float = 0.025
    stationary_bias_sigma_mps2: float = 0.08
    stationary_accel_threshold_mps2: float = 0.30
    stationary_gyro_threshold_dps: float = 1.20
    stationary_hold_s: float = 0.22
    # Salt IMU ile elde taşınan kullanımda gözlenemeyen sabit hızın uzun süre
    # taşınmasını önler. Bu değer gerçek hareketi de bir miktar sönümler.
    velocity_leak_per_s: float = 0.25
    minimum_speed_for_distance_mps: float = 0.03
    maximum_acceleration_mps2: float = 35.0
    minimum_dt_s: float = 0.001
    maximum_dt_s: float = 0.025


@dataclass(slots=True, frozen=True)
class NavSolution:
    timestamp_us: int
    position_m: FloatArray
    velocity_mps: FloatArray
    acceleration_world_mps2: FloatArray
    acceleration_bias_mps2: FloatArray
    stationary: bool
    valid: bool
    speed_mps: float
    distance_m: float


class StationaryDetector:
    def __init__(self, config: NavConfig) -> None:
        self.config = config
        self.stationary = False
        self.candidate = False
        self._candidate_time_s = 0.0

    def reset(self) -> None:
        self.stationary = False
        self.candidate = False
        self._candidate_time_s = 0.0

    def update(self, accel_norm: float, gyro_norm: float, dt: float) -> bool:
        accel_limit = self.config.stationary_accel_threshold_mps2
        gyro_limit = self.config.stationary_gyro_threshold_dps

        if self.stationary:
            still_stationary = (
                accel_norm <= accel_limit * 1.8
                and gyro_norm <= gyro_limit * 1.8
            )
            if not still_stationary:
                self.stationary = False
                self.candidate = False
                self._candidate_time_s = 0.0
            else:
                self.candidate = True
            return self.stationary

        candidate = accel_norm <= accel_limit and gyro_norm <= gyro_limit
        self.candidate = candidate
        if candidate:
            self._candidate_time_s += dt
            if self._candidate_time_s >= self.config.stationary_hold_s:
                self.stationary = True
        else:
            self._candidate_time_s = 0.0

        return self.stationary


class InertialNavigationEKF:
    """
    State vector: world position, world velocity, world acceleration bias.

    The firmware supplies gravity-removed world acceleration. The EKF performs
    short-term strapdown integration and constrains velocity/bias whenever a
    persistent stationary interval is detected.
    """

    def __init__(self, config: NavConfig | None = None) -> None:
        self.config = config or NavConfig()
        self.stationary_detector = StationaryDetector(self.config)
        self.state = np.zeros(9, dtype=np.float64)
        self.covariance = np.eye(9, dtype=np.float64)
        self.last_timestamp_us: int | None = None
        self.last_sequence: int | None = None
        self.distance_m = 0.0
        self.accepted_samples = 0
        self.rejected_samples = 0
        self.sequence_gaps = 0
        self.zupt_updates = 0
        self._last_position = np.zeros(3, dtype=np.float64)
        self._rest_anchor_position: FloatArray | None = None
        self._rest_anchor_distance_m = 0.0

        self.covariance[0:3, 0:3] *= 1.0
        self.covariance[3:6, 3:6] *= 0.25
        self.covariance[6:9, 6:9] *= 0.10

    @property
    def position_m(self) -> FloatArray:
        return self.state[0:3]

    @property
    def velocity_mps(self) -> FloatArray:
        return self.state[3:6]

    @property
    def acceleration_bias_mps2(self) -> FloatArray:
        return self.state[6:9]

    def reset(self, timestamp_us: int | None = None) -> None:
        self.state.fill(0.0)
        self.covariance.fill(0.0)
        self.covariance[0:3, 0:3] = np.eye(3)
        self.covariance[3:6, 3:6] = np.eye(3) * 0.25
        self.covariance[6:9, 6:9] = np.eye(3) * 0.10
        self.last_timestamp_us = timestamp_us
        self.last_sequence = None
        self.distance_m = 0.0
        self.accepted_samples = 0
        self.rejected_samples = 0
        self.sequence_gaps = 0
        self.zupt_updates = 0
        self._last_position.fill(0.0)
        self._rest_anchor_position = None
        self._rest_anchor_distance_m = 0.0
        self.stationary_detector.reset()

    def update(self, frame: ImuSampleFrame) -> NavSolution:
        dt = self._calculate_dt(frame)
        acceleration = np.asarray(
            frame.linear_accel_world_mps2, dtype=np.float64
        )
        specific_force_g = np.asarray(frame.accel_corrected_g, dtype=np.float64)
        gyro = np.asarray(frame.gyro_corrected_dps, dtype=np.float64)

        required = (
            StatusFlag.ACCEL_OK
            | StatusFlag.GYRO_OK
            | StatusFlag.IMU_CALIBRATED
            | StatusFlag.AHRS_OK
            | StatusFlag.LINEAR_ACCEL_OK
        )
        valid = (frame.status & required) == required
        finite = (
            np.all(np.isfinite(acceleration))
            and np.all(np.isfinite(specific_force_g))
            and np.all(np.isfinite(gyro))
        )
        plausible = np.linalg.norm(acceleration) <= self.config.maximum_acceleration_mps2

        if not valid or not finite or not plausible or dt is None:
            self.rejected_samples += 1
            self.stationary_detector.reset()
            self._rest_anchor_position = None
            return self._solution(frame, acceleration, False)

        # Durağanlık için gravity-removal çıktısını kullanmak güvenilir değildir:
        # küçük bir AHRS eğim hatası g'nin bir bölümünü doğrusal ivme gibi üretir.
        # Sabit bir IMU'da yönü ne olursa olsun özgül kuvvet büyüklüğü yaklaşık 1 g'dir.
        rest_accel_error_mps2 = (
            abs(float(np.linalg.norm(specific_force_g)) - 1.0) * 9.80665
        )
        stationary = self.stationary_detector.update(
            rest_accel_error_mps2,
            float(np.linalg.norm(gyro)),
            dt,
        )

        if self.stationary_detector.candidate:
            if self._rest_anchor_position is None:
                self._rest_anchor_position = self.state[0:3].copy()
                self._rest_anchor_distance_m = self.distance_m
        else:
            self._rest_anchor_position = None

        self._predict(acceleration, dt)

        if stationary:
            self._zero_velocity_update()
            self._stationary_bias_update(acceleration)
            self.state[3:6] *= 0.05
            # Duruş kararının verilmesi için geçen hold süresinde oluşan sahte
            # hareketi de geri al; aksi halde her dur-kalk mesafeyi şişirir.
            if self._rest_anchor_position is not None:
                self.state[0:3] = self._rest_anchor_position
                self.distance_m = self._rest_anchor_distance_m
            self.zupt_updates += 1
        elif self.config.velocity_leak_per_s > 0.0:
            damping = math.exp(-self.config.velocity_leak_per_s * dt)
            self.state[3:6] *= damping

        displacement = self.state[0:3] - self._last_position
        if (
            not stationary
            and float(np.linalg.norm(self.state[3:6]))
            >= self.config.minimum_speed_for_distance_mps
        ):
            self.distance_m += float(np.linalg.norm(displacement))
        self._last_position = self.state[0:3].copy()
        self.accepted_samples += 1

        return self._solution(frame, acceleration, True)

    def _calculate_dt(self, frame: ImuSampleFrame) -> float | None:
        sequence = frame.header.sequence
        if self.last_sequence is not None:
            expected = (self.last_sequence + 1) & 0xFFFFFFFF
            if sequence != expected:
                self.sequence_gaps += min(
                    (sequence - expected) & 0xFFFFFFFF, 1_000_000
                )
        self.last_sequence = sequence

        timestamp_us = frame.header.timestamp_us
        timestamp_dt: float | None = None
        if self.last_timestamp_us is not None and timestamp_us > self.last_timestamp_us:
            timestamp_dt = (timestamp_us - self.last_timestamp_us) * 1.0e-6
        self.last_timestamp_us = timestamp_us

        dt = timestamp_dt if timestamp_dt is not None else frame.delta_time_s
        if not math.isfinite(dt):
            return None

        if not (self.config.minimum_dt_s <= dt <= self.config.maximum_dt_s):
            payload_dt = frame.delta_time_s
            if self.config.minimum_dt_s <= payload_dt <= self.config.maximum_dt_s:
                return float(payload_dt)
            return None

        return float(dt)

    def _predict(self, measured_acceleration: FloatArray, dt: float) -> None:
        acceleration = measured_acceleration - self.state[6:9]
        dt2 = dt * dt

        self.state[0:3] += self.state[3:6] * dt + 0.5 * acceleration * dt2
        self.state[3:6] += acceleration * dt

        identity = np.eye(3, dtype=np.float64)
        transition = np.eye(9, dtype=np.float64)
        transition[0:3, 3:6] = identity * dt
        transition[0:3, 6:9] = -identity * (0.5 * dt2)
        transition[3:6, 6:9] = -identity * dt

        noise_map = np.zeros((9, 6), dtype=np.float64)
        noise_map[0:3, 0:3] = identity * (0.5 * dt2)
        noise_map[3:6, 0:3] = identity * dt
        noise_map[6:9, 3:6] = identity * math.sqrt(dt)

        noise = np.diag(
            [self.config.acceleration_noise_mps2**2] * 3
            + [self.config.accel_bias_random_walk_mps3**2] * 3
        )

        self.covariance = (
            transition @ self.covariance @ transition.T
            + noise_map @ noise @ noise_map.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def _zero_velocity_update(self) -> None:
        measurement = np.zeros(3, dtype=np.float64)
        observation = np.zeros((3, 9), dtype=np.float64)
        observation[:, 3:6] = np.eye(3)
        variance = self.config.zupt_velocity_sigma_mps**2
        self._measurement_update(measurement, observation, np.eye(3) * variance)

    def _stationary_bias_update(self, measured_acceleration: FloatArray) -> None:
        observation = np.zeros((3, 9), dtype=np.float64)
        observation[:, 6:9] = np.eye(3)
        variance = self.config.stationary_bias_sigma_mps2**2
        self._measurement_update(
            measured_acceleration,
            observation,
            np.eye(3) * variance,
        )

    def _measurement_update(
        self,
        measurement: FloatArray,
        observation: FloatArray,
        measurement_noise: FloatArray,
    ) -> None:
        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_noise
        )
        gain = np.linalg.solve(
            innovation_covariance,
            observation @ self.covariance,
        ).T

        self.state += gain @ innovation

        identity = np.eye(9, dtype=np.float64)
        residual = identity - gain @ observation
        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ measurement_noise @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def _solution(
        self,
        frame: ImuSampleFrame,
        acceleration: FloatArray,
        valid: bool,
    ) -> NavSolution:
        velocity = self.state[3:6].copy()
        return NavSolution(
            timestamp_us=frame.header.timestamp_us,
            position_m=self.state[0:3].copy(),
            velocity_mps=velocity,
            acceleration_world_mps2=acceleration.copy(),
            acceleration_bias_mps2=self.state[6:9].copy(),
            stationary=self.stationary_detector.stationary,
            valid=valid,
            speed_mps=float(np.linalg.norm(velocity)),
            distance_m=self.distance_m,
        )
