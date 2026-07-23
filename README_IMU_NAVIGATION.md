# ESP32 / GY-85 Real-Time Inertial Navigation

This application decodes the firmware's 2 Mbaud binary stream, validates every
CRC32 frame, runs a 9-state inertial-navigation EKF, applies persistent
stationary detection and zero-velocity updates (ZUPT), and renders the resulting
route in live 3D and top-down views.

## Installation

```bash
cd "/Users/serkantanis/Documents/New project"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run

List likely ports on macOS:

```bash
ls /dev/cu.usb* /dev/cu.SLAB* 2>/dev/null
```

Start with a port selected in the UI:

```bash
python3 imu_navigation.py
```

Start and connect immediately:

```bash
python3 imu_navigation.py --port /dev/cu.usbmodem1101
```

Run without hardware using the synthetic 200 Hz binary source:

```bash
python3 imu_navigation.py --demo
```

## Raspberry Pi / browser status screen

The web mode is headless and does not need PySide, pyqtgraph, or OpenGL:

```bash
cd /home/pi/imu-navigation
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-rpi.txt
python3 imu_navigation.py --web --port /dev/ttyUSB0
```

For an ESP32 that appears as an ACM serial device, use `/dev/ttyACM0`. If
`--port` is omitted, the first USB/ACM serial device is selected automatically.
The web screen also starts camera 0 with `rpicam-vid` and embeds its MJPEG stream.
The default is 1280x720 at 30 FPS:

```bash
python3 imu_navigation.py --web --port /dev/ttyACM0 \
  --camera-width 1280 --camera-height 720 --camera-fps 30
```

Use `--camera-quality 70` to reduce network traffic, or `--no-camera` to run
only the IMU dashboard. Camera failure does not stop telemetry.

Open the address printed by the program from a phone or computer on the same
network, normally:

```text
http://raspberrypi.local:8080
```

Test the browser UI without hardware:

```bash
python3 imu_navigation.py --web --demo
```

The server listens on all network interfaces by default. Change the TCP port
with `--http-port 8081`. Do not forward this port to the public internet; the
dashboard is intended for a trusted local network.

## Operating procedure

1. Place the IMU still and Z-up during its initial calibration.
2. Rotate it through all axes until magnetometer calibration reaches 100%.
3. Put it still for at least one second before pressing **Rotayı sıfırla**.
4. Begin motion. Short stationary pauses let ZUPT remove accumulated velocity
   error.
5. Export the current route and telemetry with **CSV dışa aktar**.

## Accuracy boundary

An IMU alone cannot produce a globally stable long-term navigation route.
Accelerometer bias is integrated twice, so small residual errors become large
position drift. This implementation is optimized for short-duration and
stop-and-go motion using an acceleration-bias EKF, ZUPT, outlier rejection,
sequence-gap monitoring, and optional gentle velocity leakage.

For absolute or long-duration navigation, fuse at least one external position
or velocity source such as GNSS, UWB anchors, wheel odometry, optical flow, or
visual-inertial odometry. The firmware preintegration frames are decoded and
displayed, but the desktop EKF uses the 200 Hz gravity-removed sample frames to
avoid double-integrating the same measurement.
