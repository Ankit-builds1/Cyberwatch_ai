"""Docker entry point for the terminal-only CyberWatch distribution."""

import os
import sys

import download_models


LIVE_ERROR = (
    "The live command is not supported inside Docker Desktop. "
    "Run live capture natively on Windows with Python, Npcap, and an "
    "Administrator terminal."
)


def main(args=None):
    command_args = list(sys.argv[1:] if args is None else args)
    if command_args and command_args[0] == "live":
        raise SystemExit(LIVE_ERROR)
    if not command_args:
        command_args = ["info"]
    download_models.main()
    os.execv(sys.executable, [sys.executable, "predict.py", *command_args])


if __name__ == "__main__":
    main()
