"""Decoder and test encoder for the ESP32 GY-85 binary protocol v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
import struct
import zlib
from typing import Iterable


MAGIC = 0xA55A
MAGIC_BYTES = b"\x5a\xa5"
VERSION = 2
MAX_PAYLOAD_LENGTH = 4096

HEADER = struct.Struct("<HBBHIQ")
CRC = struct.Struct("<I")
IMU_PAYLOAD = struct.Struct("<HH32f")
PREINTEGRATION_PAYLOAD = struct.Struct("<fHH16f")


class FrameType(IntEnum):
    IMU_SAMPLE = 1
    PREINTEGRATION = 2


class StatusFlag(IntFlag):
    ACCEL_OK = 1 << 0
    GYRO_OK = 1 << 1
    MAG_OK = 1 << 2
    IMU_CALIBRATED = 1 << 3
    MAG_CALIBRATED = 1 << 4
    AHRS_OK = 1 << 5
    AHRS_USES_MAG = 1 << 6
    LINEAR_ACCEL_OK = 1 << 7
    DYNAMIC_BIAS_ACTIVE = 1 << 8
    ACCEL_SIX_FACE_CALIBRATED = 1 << 9


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(slots=True, frozen=True)
class FrameHeader:
    version: int
    frame_type: int
    payload_length: int
    sequence: int
    timestamp_us: int


@dataclass(slots=True, frozen=True)
class ImuSampleFrame:
    header: FrameHeader
    status: StatusFlag
    delta_time_s: float
    accel_raw_g: Vector3
    accel_corrected_g: Vector3
    gyro_raw_dps: Vector3
    gyro_corrected_dps: Vector3
    gyro_bias_applied_dps: Vector3
    mag_gauss: Vector3
    temperature_c: float
    quaternion_wxyz: Quaternion
    euler_deg: Vector3
    linear_accel_world_mps2: Vector3
    calibration_progress: float
    mag_calibration_progress: float


@dataclass(slots=True, frozen=True)
class PreintegrationFrame:
    header: FrameHeader
    integration_time_s: float
    sample_count: int
    status: StatusFlag
    delta_quaternion_wxyz: Quaternion
    delta_velocity_mps: Vector3
    delta_position_m: Vector3
    gyro_bias_applied_dps: Vector3
    accel_bias_g: Vector3


DecodedFrame = ImuSampleFrame | PreintegrationFrame


@dataclass(slots=True)
class ParserStats:
    valid_frames: int = 0
    crc_errors: int = 0
    malformed_frames: int = 0
    discarded_bytes: int = 0
    sequence_gaps: int = 0
    frames_by_type: dict[int, int] = field(default_factory=dict)


def _vec(values: Iterable[float]) -> Vector3:
    x, y, z = values
    return float(x), float(y), float(z)


def _quat(values: Iterable[float]) -> Quaternion:
    w, x, y, z = values
    return float(w), float(x), float(y), float(z)


class BinaryProtocolParser:
    """Incremental, resynchronizing CRC-protected frame parser."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.stats = ParserStats()
        self._last_imu_sequence: int | None = None

    def reset(self) -> None:
        self.buffer.clear()
        self.stats = ParserStats()
        self._last_imu_sequence = None

    def feed(self, data: bytes | bytearray | memoryview) -> list[DecodedFrame]:
        if data:
            self.buffer.extend(data)

        decoded: list[DecodedFrame] = []

        while True:
            if len(self.buffer) < HEADER.size:
                break

            magic_index = self.buffer.find(MAGIC_BYTES)
            if magic_index < 0:
                keep = 1 if self.buffer[-1:] == MAGIC_BYTES[:1] else 0
                discarded = len(self.buffer) - keep
                if discarded:
                    del self.buffer[:discarded]
                    self.stats.discarded_bytes += discarded
                break

            if magic_index:
                del self.buffer[:magic_index]
                self.stats.discarded_bytes += magic_index

            if len(self.buffer) < HEADER.size:
                break

            magic, version, frame_type, payload_length, sequence, timestamp_us = (
                HEADER.unpack_from(self.buffer)
            )

            if (
                magic != MAGIC
                or version != VERSION
                or payload_length > MAX_PAYLOAD_LENGTH
            ):
                del self.buffer[0]
                self.stats.malformed_frames += 1
                continue

            total_length = HEADER.size + payload_length + CRC.size
            if len(self.buffer) < total_length:
                break

            packet_without_crc = memoryview(self.buffer)[: HEADER.size + payload_length]
            transmitted_crc = CRC.unpack_from(
                self.buffer, HEADER.size + payload_length
            )[0]
            calculated_crc = zlib.crc32(packet_without_crc) & 0xFFFFFFFF

            if transmitted_crc != calculated_crc:
                del packet_without_crc
                del self.buffer[0]
                self.stats.crc_errors += 1
                continue

            payload = bytes(self.buffer[HEADER.size : HEADER.size + payload_length])
            del packet_without_crc
            del self.buffer[:total_length]

            header = FrameHeader(
                version=version,
                frame_type=frame_type,
                payload_length=payload_length,
                sequence=sequence,
                timestamp_us=timestamp_us,
            )

            try:
                frame = self._decode_payload(header, payload)
            except (ValueError, struct.error):
                self.stats.malformed_frames += 1
                continue

            self.stats.valid_frames += 1
            self.stats.frames_by_type[frame_type] = (
                self.stats.frames_by_type.get(frame_type, 0) + 1
            )

            if isinstance(frame, ImuSampleFrame):
                if self._last_imu_sequence is not None:
                    expected = (self._last_imu_sequence + 1) & 0xFFFFFFFF
                    if sequence != expected:
                        gap = (sequence - expected) & 0xFFFFFFFF
                        self.stats.sequence_gaps += min(gap, 1_000_000)
                self._last_imu_sequence = sequence

            decoded.append(frame)

        return decoded

    @staticmethod
    def _decode_payload(header: FrameHeader, payload: bytes) -> DecodedFrame:
        if header.frame_type == FrameType.IMU_SAMPLE:
            if len(payload) != IMU_PAYLOAD.size:
                raise ValueError("wrong IMU payload size")

            unpacked = IMU_PAYLOAD.unpack(payload)
            status = StatusFlag(unpacked[0])
            values = unpacked[2:]

            return ImuSampleFrame(
                header=header,
                status=status,
                delta_time_s=values[0],
                accel_raw_g=_vec(values[1:4]),
                accel_corrected_g=_vec(values[4:7]),
                gyro_raw_dps=_vec(values[7:10]),
                gyro_corrected_dps=_vec(values[10:13]),
                gyro_bias_applied_dps=_vec(values[13:16]),
                mag_gauss=_vec(values[16:19]),
                temperature_c=values[19],
                quaternion_wxyz=_quat(values[20:24]),
                euler_deg=_vec(values[24:27]),
                linear_accel_world_mps2=_vec(values[27:30]),
                calibration_progress=values[30],
                mag_calibration_progress=values[31],
            )

        if header.frame_type == FrameType.PREINTEGRATION:
            if len(payload) != PREINTEGRATION_PAYLOAD.size:
                raise ValueError("wrong preintegration payload size")

            unpacked = PREINTEGRATION_PAYLOAD.unpack(payload)
            values = unpacked[3:]

            return PreintegrationFrame(
                header=header,
                integration_time_s=unpacked[0],
                sample_count=unpacked[1],
                status=StatusFlag(unpacked[2]),
                delta_quaternion_wxyz=_quat(values[0:4]),
                delta_velocity_mps=_vec(values[4:7]),
                delta_position_m=_vec(values[7:10]),
                gyro_bias_applied_dps=_vec(values[10:13]),
                accel_bias_g=_vec(values[13:16]),
            )

        raise ValueError(f"unsupported frame type {header.frame_type}")


def encode_frame(
    frame_type: FrameType,
    sequence: int,
    timestamp_us: int,
    payload: bytes,
) -> bytes:
    """Encoder used by demo mode and tests; firmware is the source of truth."""

    header = HEADER.pack(
        MAGIC,
        VERSION,
        int(frame_type),
        len(payload),
        sequence & 0xFFFFFFFF,
        timestamp_us,
    )
    without_crc = header + payload
    return without_crc + CRC.pack(zlib.crc32(without_crc) & 0xFFFFFFFF)


def encode_imu_sample(frame: ImuSampleFrame) -> bytes:
    values = (
        frame.delta_time_s,
        *frame.accel_raw_g,
        *frame.accel_corrected_g,
        *frame.gyro_raw_dps,
        *frame.gyro_corrected_dps,
        *frame.gyro_bias_applied_dps,
        *frame.mag_gauss,
        frame.temperature_c,
        *frame.quaternion_wxyz,
        *frame.euler_deg,
        *frame.linear_accel_world_mps2,
        frame.calibration_progress,
        frame.mag_calibration_progress,
    )
    payload = IMU_PAYLOAD.pack(int(frame.status), 0, *values)
    return encode_frame(
        FrameType.IMU_SAMPLE,
        frame.header.sequence,
        frame.header.timestamp_us,
        payload,
    )
