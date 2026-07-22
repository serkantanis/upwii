from __future__ import annotations

import unittest

from imu_nav.protocol import (
    BinaryProtocolParser,
    FrameHeader,
    FrameType,
    ImuSampleFrame,
    StatusFlag,
    encode_imu_sample,
)


def sample_frame(sequence: int = 7) -> ImuSampleFrame:
    return ImuSampleFrame(
        header=FrameHeader(2, FrameType.IMU_SAMPLE, 132, sequence, 123_456),
        status=StatusFlag.ACCEL_OK | StatusFlag.GYRO_OK,
        delta_time_s=0.005,
        accel_raw_g=(0.0, 0.0, 1.0),
        accel_corrected_g=(0.0, 0.0, 1.0),
        gyro_raw_dps=(0.0, 0.0, 0.0),
        gyro_corrected_dps=(0.0, 0.0, 0.0),
        gyro_bias_applied_dps=(0.1, 0.2, 0.3),
        mag_gauss=(0.2, 0.1, 0.4),
        temperature_c=31.5,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        euler_deg=(1.0, 2.0, 3.0),
        linear_accel_world_mps2=(0.4, 0.5, 0.6),
        calibration_progress=1.0,
        mag_calibration_progress=0.75,
    )


class ProtocolTests(unittest.TestCase):
    def test_fragmented_round_trip_and_resync(self) -> None:
        packet = encode_imu_sample(sample_frame())
        parser = BinaryProtocolParser()
        decoded = []
        stream = b"garbage" + packet
        for index in range(0, len(stream), 11):
            decoded.extend(parser.feed(stream[index : index + 11]))

        self.assertEqual(len(decoded), 1)
        frame = decoded[0]
        self.assertIsInstance(frame, ImuSampleFrame)
        self.assertEqual(frame.header.sequence, 7)
        self.assertAlmostEqual(frame.temperature_c, 31.5)
        self.assertEqual(parser.stats.discarded_bytes, 7)

    def test_crc_failure_is_rejected(self) -> None:
        packet = bytearray(encode_imu_sample(sample_frame()))
        packet[40] ^= 0x80
        parser = BinaryProtocolParser()
        self.assertEqual(parser.feed(packet), [])
        self.assertEqual(parser.stats.crc_errors, 1)


if __name__ == "__main__":
    unittest.main()
