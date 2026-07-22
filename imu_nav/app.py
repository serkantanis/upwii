"""PySide6/pyqtgraph desktop UI for live inertial navigation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

from .navigation import InertialNavigationEKF, NavConfig, NavSolution
from .protocol import (
    BinaryProtocolParser,
    FrameHeader,
    FrameType,
    ImuSampleFrame,
    PreintegrationFrame,
    StatusFlag,
    encode_imu_sample,
)


EventQueue = queue.Queue[tuple[str, Any]]


@dataclass(slots=True, frozen=True)
class LogRecord:
    timestamp_us: int
    sequence: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    euler: tuple[float, float, float]
    temperature_c: float
    status: int
    stationary: bool


class SerialWorker(threading.Thread):
    def __init__(
        self,
        port: str,
        baud: int,
        events: EventQueue,
    ) -> None:
        super().__init__(daemon=True, name="imu-serial-reader")
        self.port = port
        self.baud = baud
        self.events = events
        self.parser = BinaryProtocolParser()
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            import serial

            with serial.Serial(
                self.port,
                self.baud,
                timeout=0.05,
                write_timeout=0.25,
            ) as connection:
                connection.reset_input_buffer()
                self.events.put(("status", f"Bağlandı: {self.port}"))

                while not self.stop_event.is_set():
                    data = connection.read(max(connection.in_waiting, 1))
                    for frame in self.parser.feed(data):
                        self._put_frame(frame)

                stats = self.parser.stats
                self.events.put(
                    (
                        "parser_stats",
                        (
                            stats.valid_frames,
                            stats.crc_errors,
                            stats.malformed_frames,
                            stats.sequence_gaps,
                            stats.discarded_bytes,
                        ),
                    )
                )
        except Exception as exc:  # serial exceptions vary by platform
            self.events.put(("error", f"Seri bağlantı hatası: {exc}"))
        finally:
            self.events.put(("stopped", None))

    def _put_frame(self, frame: ImuSampleFrame | PreintegrationFrame) -> None:
        try:
            self.events.put_nowait(("frame", frame))
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(("frame", frame))


class DemoWorker(threading.Thread):
    """Synthetic byte-stream source that exercises the real protocol parser."""

    def __init__(self, events: EventQueue) -> None:
        super().__init__(daemon=True, name="imu-demo-source")
        self.events = events
        self.stop_event = threading.Event()
        self.parser = BinaryProtocolParser()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.events.put(("status", "Demo veri akışı"))
        sequence = 0
        timestamp_us = 0
        next_tick = time.monotonic()

        while not self.stop_event.is_set():
            acceleration = self._demo_acceleration(timestamp_us * 1.0e-6)
            frame = self._make_frame(sequence, timestamp_us, acceleration)
            packet = encode_imu_sample(frame)

            # Deliberately fragment packets like a real serial stream.
            split = 37 + (sequence % 41)
            for chunk in (packet[:split], packet[split:]):
                for decoded in self.parser.feed(chunk):
                    self.events.put(("frame", decoded))

            sequence = (sequence + 1) & 0xFFFFFFFF
            timestamp_us += 5000
            next_tick += 0.005
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                next_tick = time.monotonic()

        self.events.put(("stopped", None))

    @staticmethod
    def _demo_acceleration(time_s: float) -> tuple[float, float, float]:
        phase = time_s % 18.0
        if 2.0 <= phase < 3.0:
            return 0.80, 0.0, 0.0
        if 6.0 <= phase < 7.0:
            return -0.80, 0.0, 0.0
        if 9.0 <= phase < 10.0:
            return 0.0, 0.65, 0.0
        if 13.0 <= phase < 14.0:
            return 0.0, -0.65, 0.0
        if 15.0 <= phase < 15.8:
            return 0.0, 0.0, 0.35
        if 16.6 <= phase < 17.4:
            return 0.0, 0.0, -0.35
        return 0.0, 0.0, 0.0

    @staticmethod
    def _make_frame(
        sequence: int,
        timestamp_us: int,
        acceleration: tuple[float, float, float],
    ) -> ImuSampleFrame:
        gravity = 9.80665
        accel_g = (
            acceleration[0] / gravity,
            acceleration[1] / gravity,
            1.0 + acceleration[2] / gravity,
        )
        status = (
            StatusFlag.ACCEL_OK
            | StatusFlag.GYRO_OK
            | StatusFlag.MAG_OK
            | StatusFlag.IMU_CALIBRATED
            | StatusFlag.MAG_CALIBRATED
            | StatusFlag.AHRS_OK
            | StatusFlag.AHRS_USES_MAG
            | StatusFlag.LINEAR_ACCEL_OK
        )
        return ImuSampleFrame(
            header=FrameHeader(
                version=2,
                frame_type=FrameType.IMU_SAMPLE,
                payload_length=132,
                sequence=sequence,
                timestamp_us=timestamp_us,
            ),
            status=status,
            delta_time_s=0.005,
            accel_raw_g=accel_g,
            accel_corrected_g=accel_g,
            gyro_raw_dps=(0.0, 0.0, 0.0),
            gyro_corrected_dps=(0.0, 0.0, 0.0),
            gyro_bias_applied_dps=(0.0, 0.0, 0.0),
            mag_gauss=(0.25, 0.0, 0.35),
            temperature_c=32.0 + 0.5 * math.sin(timestamp_us * 1.0e-7),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            euler_deg=(0.0, 0.0, 0.0),
            linear_accel_world_mps2=acceleration,
            calibration_progress=1.0,
            mag_calibration_progress=1.0,
        )


class NavigationWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        initial_port: str | None,
        baud: int,
        demo: bool,
    ) -> None:
        super().__init__()
        self.setWindowTitle("ESP32 IMU — Gerçek Zamanlı Navigasyon")
        self.resize(1480, 900)

        pg.setConfigOptions(antialias=True, foreground="w", background="#10141b")

        self.baud = baud
        self.events: EventQueue = queue.Queue(maxsize=10_000)
        self.worker: SerialWorker | DemoWorker | None = None
        self.navigator = InertialNavigationEKF()
        self.last_sample: ImuSampleFrame | None = None
        self.last_solution: NavSolution | None = None
        self.last_preintegration: PreintegrationFrame | None = None
        self.parser_stats = (0, 0, 0, 0, 0)

        self.route_points: deque[np.ndarray] = deque(maxlen=30_000)
        self.log_records: deque[LogRecord] = deque(maxlen=500_000)
        self.time_history: deque[float] = deque(maxlen=6000)
        self.speed_history: deque[float] = deque(maxlen=6000)
        self.accel_history: deque[float] = deque(maxlen=6000)
        self.frame_arrivals: deque[float] = deque(maxlen=500)
        self.plot_decimation = 4
        self.sample_counter = 0
        self.visuals_dirty = False

        self._build_ui()
        self.refresh_ports(initial_port)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._poll_events)
        self.timer.start()

        if demo:
            QtCore.QTimer.singleShot(0, self.start_demo)
        elif initial_port:
            QtCore.QTimer.singleShot(0, self.toggle_connection)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(340)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        root.addWidget(controls)

        connection_group = QtWidgets.QGroupBox("Bağlantı")
        connection_layout = QtWidgets.QGridLayout(connection_group)
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Yenile")
        self.connect_button = QtWidgets.QPushButton("Bağlan")
        self.demo_button = QtWidgets.QPushButton("Demo")
        connection_layout.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        connection_layout.addWidget(self.port_combo, 0, 1, 1, 2)
        connection_layout.addWidget(self.refresh_button, 1, 0)
        connection_layout.addWidget(self.connect_button, 1, 1)
        connection_layout.addWidget(self.demo_button, 1, 2)
        controls_layout.addWidget(connection_group)

        self.refresh_button.clicked.connect(lambda: self.refresh_ports(None))
        self.connect_button.clicked.connect(self.toggle_connection)
        self.demo_button.clicked.connect(self.start_demo)

        filter_group = QtWidgets.QGroupBox("Navigasyon filtresi")
        filter_form = QtWidgets.QFormLayout(filter_group)
        self.accel_threshold = self._spinbox(0.05, 3.0, 0.05, 0.30)
        self.gyro_threshold = self._spinbox(0.1, 20.0, 0.1, 1.20)
        self.velocity_leak = self._spinbox(0.0, 2.0, 0.025, 0.25, 3)
        filter_form.addRow("Durağan accel (m/s²)", self.accel_threshold)
        filter_form.addRow("Durağan gyro (°/s)", self.gyro_threshold)
        filter_form.addRow("Hız sönümü (1/s)", self.velocity_leak)
        controls_layout.addWidget(filter_group)

        for spinbox in (
            self.accel_threshold,
            self.gyro_threshold,
            self.velocity_leak,
        ):
            spinbox.valueChanged.connect(self._apply_filter_controls)

        status_group = QtWidgets.QGroupBox("Canlı durum")
        status_form = QtWidgets.QFormLayout(status_group)
        self.status_label = QtWidgets.QLabel("Bekliyor")
        self.rate_label = QtWidgets.QLabel("0.0 Hz")
        self.position_label = QtWidgets.QLabel("0, 0, 0 m")
        self.velocity_label = QtWidgets.QLabel("0, 0, 0 m/s")
        self.euler_label = QtWidgets.QLabel("0, 0, 0°")
        self.temperature_label = QtWidgets.QLabel("—")
        self.raw_accel_label = QtWidgets.QLabel("—")
        self.raw_gyro_label = QtWidgets.QLabel("—")
        self.calibration_blocker_label = QtWidgets.QLabel("—")
        self.calibration_blocker_label.setWordWrap(True)
        self.distance_label = QtWidgets.QLabel("0.00 m")
        self.quality_label = QtWidgets.QLabel("—")
        self.preintegration_label = QtWidgets.QLabel("—")
        self.parser_label = QtWidgets.QLabel("CRC 0 · gap 0")
        status_form.addRow("Akış", self.status_label)
        status_form.addRow("Örnek hızı", self.rate_label)
        status_form.addRow("Konum", self.position_label)
        status_form.addRow("Hız", self.velocity_label)
        status_form.addRow("Roll/Pitch/Yaw", self.euler_label)
        status_form.addRow("Sıcaklık", self.temperature_label)
        status_form.addRow("Ham accel", self.raw_accel_label)
        status_form.addRow("Ham gyro", self.raw_gyro_label)
        status_form.addRow("Kalibrasyon engeli", self.calibration_blocker_label)
        status_form.addRow("Toplam rota", self.distance_label)
        status_form.addRow("Kalibrasyon", self.quality_label)
        status_form.addRow("Preintegration", self.preintegration_label)
        status_form.addRow("Protokol", self.parser_label)
        controls_layout.addWidget(status_group)

        actions = QtWidgets.QGroupBox("Kayıt")
        action_layout = QtWidgets.QHBoxLayout(actions)
        reset_button = QtWidgets.QPushButton("Rotayı sıfırla")
        export_button = QtWidgets.QPushButton("CSV dışa aktar")
        reset_button.clicked.connect(self.reset_navigation)
        export_button.clicked.connect(self.export_csv)
        action_layout.addWidget(reset_button)
        action_layout.addWidget(export_button)
        controls_layout.addWidget(actions)
        controls_layout.addStretch(1)

        visual_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        root.addWidget(visual_splitter, 1)

        route_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        visual_splitter.addWidget(route_splitter)

        self.view_3d = gl.GLViewWidget()
        self.view_3d.setCameraPosition(distance=10, elevation=28, azimuth=45)
        grid = gl.GLGridItem()
        grid.setSize(20, 20)
        grid.setSpacing(1, 1)
        self.view_3d.addItem(grid)
        axis = gl.GLAxisItem()
        axis.setSize(1.5, 1.5, 1.5)
        self.view_3d.addItem(axis)
        self.route_3d = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.20, 0.78, 1.0, 1.0),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.current_point = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.72, 0.20, 1.0),
            size=10.0,
        )
        self.orientation_axes = gl.GLLinePlotItem(
            pos=np.zeros((6, 3), dtype=np.float32),
            color=np.array(
                [
                    (1, 0.25, 0.25, 1),
                    (1, 0.25, 0.25, 1),
                    (0.25, 1, 0.35, 1),
                    (0.25, 1, 0.35, 1),
                    (0.25, 0.55, 1, 1),
                    (0.25, 0.55, 1, 1),
                ],
                dtype=np.float32,
            ),
            width=3.0,
            mode="lines",
        )
        self.view_3d.addItem(self.route_3d)
        self.view_3d.addItem(self.current_point)
        self.view_3d.addItem(self.orientation_axes)
        route_splitter.addWidget(self.view_3d)

        self.top_view = pg.PlotWidget(title="Üstten görünüm — X/Y")
        self.top_view.setAspectLocked(True)
        self.top_view.showGrid(x=True, y=True, alpha=0.25)
        self.top_view.setLabel("bottom", "X", units="m")
        self.top_view.setLabel("left", "Y", units="m")
        self.route_2d = self.top_view.plot(
            pen=pg.mkPen("#32c5ff", width=2),
            symbol=None,
        )
        self.current_2d = self.top_view.plot(
            pen=None,
            symbol="o",
            symbolSize=9,
            symbolBrush="#ffb833",
        )
        route_splitter.addWidget(self.top_view)

        telemetry = pg.GraphicsLayoutWidget()
        visual_splitter.addWidget(telemetry)
        speed_plot = telemetry.addPlot(row=0, col=0, title="Hız")
        accel_plot = telemetry.addPlot(row=0, col=1, title="Doğrusal ivme")
        speed_plot.showGrid(x=True, y=True, alpha=0.2)
        accel_plot.showGrid(x=True, y=True, alpha=0.2)
        speed_plot.setLabel("left", "|v|", units="m/s")
        speed_plot.setLabel("bottom", "Zaman", units="s")
        accel_plot.setLabel("left", "|a|", units="m/s²")
        accel_plot.setLabel("bottom", "Zaman", units="s")
        self.speed_curve = speed_plot.plot(pen=pg.mkPen("#ffb833", width=2))
        self.accel_curve = accel_plot.plot(pen=pg.mkPen("#8de26d", width=2))

        visual_splitter.setSizes([650, 220])
        route_splitter.setSizes([760, 440])

    @staticmethod
    def _spinbox(
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        decimals: int = 2,
    ) -> QtWidgets.QDoubleSpinBox:
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        spinbox.setValue(value)
        return spinbox

    def refresh_ports(self, preferred: str | None) -> None:
        current = preferred or self.port_combo.currentText()
        self.port_combo.clear()
        try:
            from serial.tools import list_ports

            ports = sorted(port.device for port in list_ports.comports())
        except Exception:
            ports = []

        self.port_combo.addItems(ports)
        if current:
            if current not in ports:
                self.port_combo.addItem(current)
            self.port_combo.setCurrentText(current)

    def toggle_connection(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_worker()
            return

        port = self.port_combo.currentText().strip()
        if not port:
            QtWidgets.QMessageBox.warning(
                self,
                "Port seçilmedi",
                "ESP32 seri portunu seçin veya demo modunu kullanın.",
            )
            return

        self.reset_navigation()
        self.worker = SerialWorker(port, self.baud, self.events)
        self.worker.start()
        self.connect_button.setText("Bağlantıyı kes")
        self.status_label.setText("Bağlanıyor…")

    def start_demo(self) -> None:
        self.stop_worker()
        self.reset_navigation()
        self.worker = DemoWorker(self.events)
        self.worker.start()
        self.connect_button.setText("Durdur")

    def stop_worker(self) -> None:
        worker = self.worker
        if worker:
            worker.stop()
            worker.join(timeout=1.0)
        self.worker = None
        self.connect_button.setText("Bağlan")
        self.status_label.setText("Durduruldu")

    def reset_navigation(self) -> None:
        timestamp = self.last_sample.header.timestamp_us if self.last_sample else None
        self.navigator.reset(timestamp)
        self.route_points.clear()
        self.log_records.clear()
        self.time_history.clear()
        self.speed_history.clear()
        self.accel_history.clear()
        self.route_points.append(np.zeros(3, dtype=np.float64))
        self.visuals_dirty = True

    def _apply_filter_controls(self, *_: object) -> None:
        config = self.navigator.config
        config.stationary_accel_threshold_mps2 = self.accel_threshold.value()
        config.stationary_gyro_threshold_dps = self.gyro_threshold.value()
        config.velocity_leak_per_s = self.velocity_leak.value()

    def _poll_events(self) -> None:
        processed = 0
        while processed < 1500:
            try:
                event_type, payload = self.events.get_nowait()
            except queue.Empty:
                break

            processed += 1
            if event_type == "frame":
                if isinstance(payload, ImuSampleFrame):
                    self._process_sample(payload)
                elif isinstance(payload, PreintegrationFrame):
                    self.last_preintegration = payload
            elif event_type == "status":
                self.status_label.setText(str(payload))
            elif event_type == "error":
                self.status_label.setText(str(payload))
                self.connect_button.setText("Bağlan")
            elif event_type == "parser_stats":
                self.parser_stats = payload
            elif event_type == "stopped":
                self.connect_button.setText("Bağlan")

        if self.visuals_dirty:
            self._update_visuals()
            self.visuals_dirty = False

        self._update_status_labels()

    def _process_sample(self, sample: ImuSampleFrame) -> None:
        solution = self.navigator.update(sample)
        self.last_sample = sample
        self.last_solution = solution
        self.frame_arrivals.append(time.monotonic())
        self.sample_counter += 1

        if self.sample_counter % self.plot_decimation == 0:
            self.route_points.append(solution.position_m.copy())
            relative_time = sample.header.timestamp_us * 1.0e-6
            self.time_history.append(relative_time)
            self.speed_history.append(solution.speed_mps)
            self.accel_history.append(
                float(np.linalg.norm(solution.acceleration_world_mps2))
            )
            self.visuals_dirty = True

        self.log_records.append(
            LogRecord(
                timestamp_us=sample.header.timestamp_us,
                sequence=sample.header.sequence,
                position=tuple(float(v) for v in solution.position_m),
                velocity=tuple(float(v) for v in solution.velocity_mps),
                acceleration=tuple(
                    float(v) for v in solution.acceleration_world_mps2
                ),
                euler=sample.euler_deg,
                temperature_c=sample.temperature_c,
                status=int(sample.status),
                stationary=solution.stationary,
            )
        )

    def _update_visuals(self) -> None:
        if not self.route_points:
            return

        points = np.asarray(self.route_points, dtype=np.float32)
        self.route_3d.setData(pos=points)
        self.current_point.setData(pos=points[-1:])
        self.route_2d.setData(points[:, 0], points[:, 1])
        self.current_2d.setData([points[-1, 0]], [points[-1, 1]])

        if self.last_sample:
            axes = self._orientation_segments(
                points[-1], self.last_sample.quaternion_wxyz
            )
            self.orientation_axes.setData(pos=axes)

        if self.time_history:
            times = np.asarray(self.time_history, dtype=np.float64)
            times -= times[-1]
            self.speed_curve.setData(times, np.asarray(self.speed_history))
            self.accel_curve.setData(times, np.asarray(self.accel_history))

        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = 0.5 * (minimum + maximum)
        extent = max(float(np.max(maximum - minimum)), 2.0)
        self.view_3d.setCameraPosition(
            pos=QtGui.QVector3D(*center.tolist()),
            distance=max(5.0, extent * 2.2),
        )

    @staticmethod
    def _orientation_segments(
        origin: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ) -> np.ndarray:
        w, x, y, z = quaternion
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1.0e-9:
            rotation = np.eye(3)
        else:
            w, x, y, z = (v / norm for v in (w, x, y, z))
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float32,
            )

        scale = 0.45
        segments = []
        for axis in range(3):
            segments.extend((origin, origin + rotation[:, axis] * scale))
        return np.asarray(segments, dtype=np.float32)

    def _update_status_labels(self) -> None:
        sample = self.last_sample
        solution = self.last_solution
        if not sample or not solution:
            return

        arrivals = self.frame_arrivals
        rate = 0.0
        if len(arrivals) >= 2:
            duration = arrivals[-1] - arrivals[0]
            if duration > 0:
                rate = (len(arrivals) - 1) / duration

        p = solution.position_m
        v = solution.velocity_mps
        r, pitch, yaw = sample.euler_deg
        self.rate_label.setText(f"{rate:6.1f} Hz")
        self.position_label.setText(f"{p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f} m")
        self.velocity_label.setText(f"{v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f} m/s")
        self.euler_label.setText(f"{r:+.1f}, {pitch:+.1f}, {yaw:.1f}°")
        self.temperature_label.setText(f"{sample.temperature_c:.1f} °C")
        ax, ay, az = sample.accel_raw_g
        gx, gy, gz = sample.gyro_raw_dps
        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
        self.raw_accel_label.setText(
            f"{ax:+.3f}, {ay:+.3f}, {az:+.3f} g · |a| {accel_norm:.3f}"
        )
        self.raw_gyro_label.setText(
            f"{gx:+.2f}, {gy:+.2f}, {gz:+.2f} °/s · |g| {gyro_norm:.2f}"
        )
        self.calibration_blocker_label.setText(
            self._calibration_blocker(sample, accel_norm, gyro_norm)
        )
        self.distance_label.setText(f"{solution.distance_m:.2f} m")
        self.quality_label.setText(
            f"IMU {sample.calibration_progress * 100:.0f}% · "
            f"MAG {sample.mag_calibration_progress * 100:.0f}%"
        )

        if solution.stationary:
            self.status_label.setText("Durağan · ZUPT aktif")
        elif solution.valid:
            self.status_label.setText("Navigasyon aktif")
        else:
            self.status_label.setText("AHRS/kalibrasyon bekleniyor")

        if self.last_preintegration:
            pre = self.last_preintegration
            dv = np.linalg.norm(pre.delta_velocity_mps)
            self.preintegration_label.setText(
                f"{pre.sample_count} örnek · {pre.integration_time_s * 1000:.1f} ms · "
                f"|Δv| {dv:.3f}"
            )

        _, crc_errors, malformed, parser_gaps, discarded = self.parser_stats
        worker = self.worker
        if isinstance(worker, (SerialWorker, DemoWorker)):
            stats = worker.parser.stats
            crc_errors = stats.crc_errors
            malformed = stats.malformed_frames
            parser_gaps = stats.sequence_gaps
            discarded = stats.discarded_bytes
        self.parser_label.setText(
            f"CRC {crc_errors} · bozuk {malformed} · gap {parser_gaps} · drop {discarded}B"
        )

    @staticmethod
    def _calibration_blocker(
        sample: ImuSampleFrame,
        accel_norm: float,
        gyro_norm: float,
    ) -> str:
        if sample.calibration_progress >= 1.0:
            return "Yok"

        ax, ay, az = sample.accel_raw_g
        reasons: list[str] = []

        if not math.isfinite(accel_norm) or not math.isfinite(gyro_norm):
            reasons.append("geçersiz sayı")
        if abs(accel_norm - 1.0) > 0.35:
            reasons.append(f"|a|={accel_norm:.2f}g")
        if abs(ax) > 0.25:
            reasons.append(f"X={ax:+.2f}g")
        if abs(ay) > 0.25:
            reasons.append(f"Y={ay:+.2f}g")
        if az < 0.65:
            reasons.append(f"Z={az:+.2f}g (<+0.65)")
        if gyro_norm > 60.0:
            reasons.append(f"gyro={gyro_norm:.1f}°/s")

        if reasons:
            return "Reddediliyor: " + ", ".join(reasons)
        return "Örnek uygun; ESP32 kalibrasyonu FAILED durumunda olabilir — RESET"

    def export_csv(self) -> None:
        if not self.log_records:
            QtWidgets.QMessageBox.information(
                self, "Kayıt yok", "Dışa aktarılacak örnek bulunmuyor."
            )
            return

        default_name = str(Path.cwd() / "imu_navigation.csv")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Navigasyon kaydını dışa aktar",
            default_name,
            "CSV (*.csv)",
        )
        if not filename:
            return

        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "timestamp_us",
                    "sequence",
                    "px_m",
                    "py_m",
                    "pz_m",
                    "vx_mps",
                    "vy_mps",
                    "vz_mps",
                    "ax_mps2",
                    "ay_mps2",
                    "az_mps2",
                    "roll_deg",
                    "pitch_deg",
                    "yaw_deg",
                    "temperature_c",
                    "status_flags",
                    "stationary",
                ]
            )
            for record in self.log_records:
                writer.writerow(
                    [
                        record.timestamp_us,
                        record.sequence,
                        *record.position,
                        *record.velocity,
                        *record.acceleration,
                        *record.euler,
                        record.temperature_c,
                        record.status,
                        int(record.stationary),
                    ]
                )

        self.statusBar().showMessage(f"CSV kaydedildi: {filename}", 5000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.stop_worker()
        event.accept()


def run_gui(port: str | None, baud: int, demo: bool) -> int:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    application.setApplicationName("ESP32 IMU Navigation")
    window = NavigationWindow(port, baud, demo)
    window.show()
    return application.exec()
