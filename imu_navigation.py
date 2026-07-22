#!/usr/bin/env python3
"""Launch the real-time ESP32/GY-85 inertial navigation application."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESP32 GY-85 binary telemetry navigator"
    )
    parser.add_argument("--port", help="Serial port, for example /dev/cu.usbmodem1101")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a synthetic 200 Hz byte stream without hardware",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run the headless browser status screen instead of the desktop GUI",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Web server TCP port (default: 8080)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.web:
            from imu_nav.web import run_web

            return run_web(
                args.port,
                args.baud,
                args.demo,
                args.host,
                args.http_port,
            )

        from imu_nav.app import run_gui
    except ModuleNotFoundError as exc:
        requirements = "requirements-rpi.txt" if args.web else "requirements.txt"
        print(
            "Eksik Python paketi: "
            f"{exc.name}\nKurulum: python3 -m pip install -r {requirements}",
            file=sys.stderr,
        )
        return 2

    return run_gui(args.port, args.baud, args.demo)


if __name__ == "__main__":
    raise SystemExit(main())
