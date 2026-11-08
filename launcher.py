# -*- coding: utf-8 -*-
"""Runs the trainer (app.py) and starts it again when it asks to (exit code 3 = "updated, restart me").

The .bat starts this file so the .bat itself never has to change: cmd reads a running .bat line by line,
and rewriting it during an update (git pull) can make cmd run garbage.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESTART = 3


def main():
    args = sys.argv[1:]
    env = {**os.environ, "ONECLICK_LAUNCHER": "1"}
    while True:
        code = subprocess.call([sys.executable, str(HERE / "app.py"), *args], cwd=str(HERE), env=env)
        if code != RESTART:
            return code
        print("\n 새 버전으로 다시 켜는 중...\n", flush=True)
        args = [a for a in args if a != "--no-browser"] + ["--no-browser"]  # the open page reloads itself


if __name__ == "__main__":
    sys.exit(main())
