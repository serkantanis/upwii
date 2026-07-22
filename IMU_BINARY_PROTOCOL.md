# GY-85 IMU Binary Protocol v2

All multibyte values are little-endian. Floats use IEEE-754 binary32.
Runtime text diagnostics are disabled by default, so the serial stream contains
only binary frames after startup.

## Frame layout

| Offset | Type | Field |
|---:|---|---|
| 0 | `uint16` | Magic: `0xA55A` (wire bytes `5A A5`) |
| 2 | `uint8` | Protocol version: `2` |
| 3 | `uint8` | Frame type |
| 4 | `uint16` | Payload length |
| 6 | `uint32` | Sequence number |
| 10 | `uint64` | Timestamp in microseconds |
| 18 | bytes | Payload |
| `18 + length` | `uint32` | CRC32 |

CRC uses the standard reflected CRC-32/ISO-HDLC polynomial `0xEDB88320`,
initial value `0xFFFFFFFF`, and final XOR `0xFFFFFFFF`. It covers the complete
18-byte header followed by the payload; the CRC field itself is excluded.

## Frame type 1: IMU sample

Emitted at 200 Hz. Payload length is 132 bytes.

| Payload offset | Type | Field |
|---:|---|---|
| 0 | `uint16` | Status flags |
| 2 | `uint16` | Reserved |
| 4 | `float` | Delta time, seconds |
| 8 | `float[3]` | Raw acceleration, g |
| 20 | `float[3]` | Bias-corrected acceleration, g |
| 32 | `float[3]` | Raw gyro, degrees/s |
| 44 | `float[3]` | Bias- and temperature-corrected gyro, degrees/s |
| 56 | `float[3]` | Applied gyro bias, degrees/s |
| 68 | `float[3]` | Compensated magnetometer, gauss |
| 80 | `float` | Temperature, °C |
| 84 | `float[4]` | Quaternion `[w,x,y,z]` |
| 100 | `float[3]` | Euler `[roll,pitch,yaw]`, degrees |
| 112 | `float[3]` | Gravity-removed world acceleration, m/s² |
| 124 | `float` | Initial IMU calibration progress, 0–1 |
| 128 | `float` | Magnetometer calibration progress, 0–1 |

## Frame type 2: IMU preintegration

Emitted every 10 valid calibrated samples (nominally 20 Hz / 50 ms). Payload
length is 72 bytes. The header timestamp is the start of the integration window.

| Payload offset | Type | Field |
|---:|---|---|
| 0 | `float` | Integration duration, seconds |
| 4 | `uint16` | Integrated sample count |
| 6 | `uint16` | Status flags |
| 8 | `float[4]` | Delta quaternion `[w,x,y,z]` |
| 24 | `float[3]` | Delta velocity, m/s |
| 36 | `float[3]` | Delta position, m |
| 48 | `float[3]` | Applied gyro bias, degrees/s |
| 60 | `float[3]` | Applied accelerometer bias, g |

Preintegration uses AHRS gravity-removed world-frame acceleration. A narrow
stationary clamp prevents accelerometer noise from creating velocity while the
sensor is at rest.

## Status flags

| Bit | Mask | Meaning |
|---:|---:|---|
| 0 | `0x0001` | Accelerometer read valid |
| 1 | `0x0002` | Gyro read valid |
| 2 | `0x0004` | Magnetometer data valid |
| 3 | `0x0008` | Initial IMU calibration complete |
| 4 | `0x0010` | Magnetometer calibration complete |
| 5 | `0x0020` | AHRS valid |
| 6 | `0x0040` | AHRS currently uses magnetometer |
| 7 | `0x0080` | Gravity-removed acceleration valid |
| 8 | `0x0100` | Dynamic gyro bias updated on this sample |
| 9 | `0x0200` | Six-face accelerometer calibration complete |
