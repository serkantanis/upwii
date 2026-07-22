"""Headless local-network status screen for the IMU navigator."""

from __future__ import annotations

from collections import deque
import json
import math
import queue
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .navigation import InertialNavigationEKF
from .protocol import (
    BinaryProtocolParser,
    FrameHeader,
    FrameType,
    ImuSampleFrame,
    StatusFlag,
    encode_imu_sample,
)


EventQueue = queue.Queue[tuple[str, Any]]


class SerialWorker(threading.Thread):
    def __init__(self, port: str, baud: int, events: EventQueue) -> None:
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

            with serial.Serial(self.port, self.baud, timeout=0.05) as connection:
                connection.reset_input_buffer()
                self.events.put(("status", f"Bağlandı: {self.port}"))
                while not self.stop_event.is_set():
                    data = connection.read(max(connection.in_waiting, 1))
                    for frame in self.parser.feed(data):
                        self.events.put(("frame", frame))
        except Exception as exc:
            self.events.put(("error", f"Seri bağlantı hatası: {exc}"))
        finally:
            self.events.put(("stopped", None))


class DemoWorker(threading.Thread):
    def __init__(self, events: EventQueue) -> None:
        super().__init__(daemon=True, name="imu-demo-source")
        self.events = events
        self.parser = BinaryProtocolParser()
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.events.put(("status", "Demo veri akışı"))
        sequence = 0
        timestamp_us = 0
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            t = timestamp_us * 1.0e-6
            phase = t % 18.0
            acceleration = (0.0, 0.0, 0.0)
            if 2.0 <= phase < 3.0:
                acceleration = (0.8, 0.0, 0.0)
            elif 6.0 <= phase < 7.0:
                acceleration = (-0.8, 0.0, 0.0)
            elif 9.0 <= phase < 10.0:
                acceleration = (0.0, 0.65, 0.0)
            elif 13.0 <= phase < 14.0:
                acceleration = (0.0, -0.65, 0.0)
            frame = self._make_frame(sequence, timestamp_us, acceleration)
            for decoded in self.parser.feed(encode_imu_sample(frame)):
                self.events.put(("frame", decoded))
            sequence = (sequence + 1) & 0xFFFFFFFF
            timestamp_us += 5000
            next_tick += 0.005
            time.sleep(max(0.0, next_tick - time.monotonic()))
        self.events.put(("stopped", None))

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
            header=FrameHeader(2, FrameType.IMU_SAMPLE, 132, sequence, timestamp_us),
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


class WebState:
    def __init__(self, worker: SerialWorker | DemoWorker) -> None:
        self.worker = worker
        self.events: EventQueue = worker.events
        self.navigator = InertialNavigationEKF()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.status = "Başlatılıyor"
        self.last_sample: ImuSampleFrame | None = None
        self.last_solution = None
        self.arrivals: deque[float] = deque(maxlen=500)
        self.route: deque[list[float]] = deque(maxlen=2500)
        self.speed: deque[list[float]] = deque(maxlen=500)
        self.accel: deque[list[float]] = deque(maxlen=500)
        self.sample_counter = 0

    def run(self) -> None:
        self.worker.start()
        while not self.stop_event.is_set():
            try:
                event_type, payload = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                if event_type == "frame" and isinstance(payload, ImuSampleFrame):
                    self._process(payload)
                elif event_type in ("status", "error"):
                    self.status = str(payload)
                elif event_type == "stopped" and not self.stop_event.is_set():
                    self.status = "Veri akışı durdu"

    def _process(self, sample: ImuSampleFrame) -> None:
        solution = self.navigator.update(sample)
        self.last_sample = sample
        self.last_solution = solution
        self.arrivals.append(time.monotonic())
        self.sample_counter += 1
        if solution.stationary:
            self.status = "Durağan · ZUPT aktif"
        elif solution.valid:
            self.status = "Navigasyon aktif"
        else:
            self.status = "AHRS/kalibrasyon bekleniyor"
        if self.sample_counter % 4 == 0:
            t = sample.header.timestamp_us * 1.0e-6
            self.route.append([float(x) for x in solution.position_m])
            self.speed.append([t, solution.speed_mps])
            self.accel.append(
                [t, float(np.linalg.norm(solution.acceleration_world_mps2))]
            )

    def reset(self) -> None:
        with self.lock:
            timestamp = self.last_sample.header.timestamp_us if self.last_sample else None
            self.navigator.reset(timestamp)
            self.route.clear()
            self.speed.clear()
            self.accel.clear()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            sample = self.last_sample
            solution = self.last_solution
            rate = 0.0
            if len(self.arrivals) >= 2:
                duration = self.arrivals[-1] - self.arrivals[0]
                if duration > 0:
                    rate = (len(self.arrivals) - 1) / duration
            stats = self.worker.parser.stats
            result: dict[str, Any] = {
                "status": self.status,
                "online": sample is not None and bool(self.arrivals)
                and time.monotonic() - self.arrivals[-1] < 2.0,
                "rate_hz": rate,
                "route": list(self.route),
                "speed_history": list(self.speed),
                "accel_history": list(self.accel),
                "parser": {
                    "crc": stats.crc_errors,
                    "malformed": stats.malformed_frames,
                    "gaps": stats.sequence_gaps,
                    "discarded": stats.discarded_bytes,
                },
            }
            if sample is None or solution is None:
                return result
            result.update(
                {
                    "position": [float(x) for x in solution.position_m],
                    "velocity": [float(x) for x in solution.velocity_mps],
                    "acceleration": [
                        float(x) for x in solution.acceleration_world_mps2
                    ],
                    "speed_mps": solution.speed_mps,
                    "distance_m": solution.distance_m,
                    "stationary": solution.stationary,
                    "euler": list(sample.euler_deg),
                    "temperature_c": sample.temperature_c,
                    "accel_raw_g": list(sample.accel_raw_g),
                    "gyro_raw_dps": list(sample.gyro_raw_dps),
                    "imu_calibration": sample.calibration_progress,
                    "mag_calibration": sample.mag_calibration_progress,
                    "sequence": sample.header.sequence,
                }
            )
            return result

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.stop()


class DashboardHandler(BaseHTTPRequestHandler):
    state: WebState

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            body = json.dumps(self.state.snapshot(), separators=(",", ":")).encode()
            self._send(HTTPStatus.OK, body, "application/json")
        elif path == "/health":
            self._send(HTTPStatus.OK, b"ok\n", "text/plain")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/api/reset":
            self.state.reset()
            self._send(HTTPStatus.NO_CONTENT, b"", "text/plain")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _detect_port() -> str | None:
    try:
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        preferred = [p.device for p in ports if "ttyACM" in p.device or "ttyUSB" in p.device]
        return preferred[0] if preferred else (ports[0].device if ports else None)
    except Exception:
        return None


def run_web(
    port: str | None,
    baud: int,
    demo: bool,
    host: str,
    http_port: int,
) -> int:
    if not 1 <= http_port <= 65535:
        print("HTTP portu 1-65535 arasında olmalı")
        return 2
    if demo:
        events: EventQueue = queue.Queue(maxsize=10_000)
        worker: SerialWorker | DemoWorker = DemoWorker(events)
    else:
        port = port or _detect_port()
        if not port:
            print("Seri port bulunamadı. --port /dev/ttyUSB0 ile belirtin.")
            return 2
        events = queue.Queue(maxsize=10_000)
        worker = SerialWorker(port, baud, events)

    state = WebState(worker)
    DashboardHandler.state = state
    processor = threading.Thread(target=state.run, daemon=True, name="imu-processor")
    processor.start()
    try:
        server = DashboardServer((host, http_port), DashboardHandler)
    except OSError as exc:
        state.stop()
        print(f"Web sunucusu başlatılamadı: {exc}")
        return 2

    print(f"IMU durum ekranı: http://{_local_ip()}:{http_port}")
    print("Durdurmak için Ctrl+C")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nDurduruluyor…")
    finally:
        server.server_close()
        state.stop()
        processor.join(timeout=1.5)
    return 0


DASHBOARD_HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IMU Navigasyon</title><style>
:root{color-scheme:dark;--bg:#071018;--panel:#101d27;--line:#233746;--cyan:#39d7ff;--amber:#ffbd4a;--green:#73e28f;--muted:#91a7b5}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 70% -20%,#173647,var(--bg) 45%);font:15px system-ui,sans-serif;color:#edf8ff}.wrap{max-width:1280px;margin:auto;padding:22px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}h1{font-size:22px;margin:0;letter-spacing:.04em}.badge{padding:7px 12px;border:1px solid var(--line);border-radius:99px;color:var(--muted)}.badge.on{color:var(--green);border-color:#285b3c}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:linear-gradient(145deg,#12232e,#0c1821);border:1px solid var(--line);border-radius:14px;padding:16px;min-width:0}.wide{grid-column:span 2}.full{grid-column:1/-1}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}.value{font:600 25px ui-monospace,monospace;margin-top:8px;white-space:nowrap}.small{font-size:17px}canvas{width:100%;height:240px;margin-top:8px;display:block}button{background:#193341;color:white;border:1px solid #335364;border-radius:9px;padding:9px 14px;cursor:pointer}.stats{color:var(--muted);font:13px ui-monospace,monospace;margin-top:10px}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.wide{grid-column:span 2}.value{font-size:20px}}@media(max-width:480px){.wrap{padding:12px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}}
</style></head><body><div class="wrap"><header><div><h1>ESP32 · IMU NAVİGASYON</h1><div id="status" class="stats">Bağlanıyor…</div></div><div id="online" class="badge">ÇEVRİMDIŞI</div></header>
<main class="grid"><section class="card"><div class="label">Örnek hızı</div><div id="rate" class="value">—</div></section><section class="card"><div class="label">Hız</div><div id="speed" class="value">—</div></section><section class="card"><div class="label">Toplam rota</div><div id="distance" class="value">—</div></section><section class="card"><div class="label">Sıcaklık</div><div id="temp" class="value">—</div></section>
<section class="card wide"><div class="label">Konum X / Y / Z</div><div id="position" class="value small">—</div></section><section class="card wide"><div class="label">Hız X / Y / Z</div><div id="velocity" class="value small">—</div></section>
<section class="card wide"><div class="label">Üstten rota · X/Y</div><canvas id="route"></canvas></section><section class="card wide"><div class="label">Hız geçmişi</div><canvas id="chart"></canvas></section>
<section class="card wide"><div class="label">Roll / Pitch / Yaw</div><div id="euler" class="value small">—</div></section><section class="card wide"><div class="label">Kalibrasyon</div><div id="cal" class="value small">—</div></section>
<section class="card full"><button onclick="resetNav()">Rotayı sıfırla</button><span id="parser" class="stats" style="margin-left:14px">—</span></section></main></div>
<script>
const $=id=>document.getElementById(id), fmt=(a,n=2,u='')=>a?a.map(x=>(x>=0?'+':'')+x.toFixed(n)).join(' / ')+u:'—';
function canvas(id,points,color,xy=false){const c=$(id),d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,w,h);x.strokeStyle='#233746';x.lineWidth=1;x.beginPath();for(let i=1;i<4;i++){x.moveTo(0,h*i/4);x.lineTo(w,h*i/4)}x.stroke();if(!points||points.length<2)return;let data=xy?points.map(p=>[p[0],p[1]]):points.map(p=>[p[0],p[1]]);let xs=data.map(p=>p[0]),ys=data.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;let pad=18;x.strokeStyle=color;x.lineWidth=2;x.beginPath();data.forEach((p,i)=>{let px=pad+(p[0]-xmin)/(xmax-xmin)*(w-pad*2),py=h-pad-(p[1]-ymin)/(ymax-ymin)*(h-pad*2);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}
async function update(){try{let r=await fetch('/api/status',{cache:'no-store'}),s=await r.json();$('online').textContent=s.online?'CANLI':'ÇEVRİMDIŞI';$('online').className='badge '+(s.online?'on':'');$('status').textContent=s.status;$('rate').textContent=s.rate_hz.toFixed(1)+' Hz';$('speed').textContent=s.speed_mps==null?'—':s.speed_mps.toFixed(2)+' m/s';$('distance').textContent=s.distance_m==null?'—':s.distance_m.toFixed(2)+' m';$('temp').textContent=s.temperature_c==null?'—':s.temperature_c.toFixed(1)+' °C';$('position').textContent=fmt(s.position,2,' m');$('velocity').textContent=fmt(s.velocity,2,' m/s');$('euler').textContent=fmt(s.euler,1,'°');$('cal').textContent=s.imu_calibration==null?'—':'IMU '+(s.imu_calibration*100).toFixed(0)+'% · MAG '+(s.mag_calibration*100).toFixed(0)+'%';$('parser').textContent=`CRC ${s.parser.crc} · bozuk ${s.parser.malformed} · gap ${s.parser.gaps} · drop ${s.parser.discarded}B`;canvas('route',s.route,'#39d7ff',true);canvas('chart',s.speed_history,'#ffbd4a')}catch(e){$('online').textContent='BAĞLANTI YOK';$('online').className='badge'}}
async function resetNav(){if(confirm('Rota ve navigasyon durumu sıfırlansın mı?'))await fetch('/api/reset',{method:'POST'})}update();setInterval(update,500);addEventListener('resize',update);
</script></body></html>"""
