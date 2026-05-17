"""
Keyboard Commander — global keyboard listener for robot play control.

Uses pynput for OS-level keyboard hooking: no window, no focus required.
Runs a daemon thread in the background, compatible with Isaac Gym viewer.

Key mapping:
    W/S   — forward/backward (lin_vel_x)
    A/D   — left/right (lin_vel_y)
    Q/E   — turn left/right (heading target)
    Space — action trigger (fire projectile)
    R     — reset trigger (reset projectiles)
    Esc   — exit play loop

Usage:
    kb = KeyboardCommander(lin_vel_x_range=(-1.0, 1.0), lin_vel_y_range=(-1.0, 1.0))
    kb.start()
    while True:
        cmd = kb.get_commands()
        # cmd = (lin_vel_x, lin_vel_y, heading_target, action, reset, quit)
        if cmd.quit:
            break
    kb.stop()

Dependency: pip install pynput
"""

import threading
from collections import namedtuple

KeyboardCommand = namedtuple(
    "KeyboardCommand",
    ["lin_vel_x", "lin_vel_y", "heading_target", "action", "reset", "quit"],
)


class KeyboardCommander:
    def __init__(
        self,
        lin_vel_x_range=(-1.0, 1.0),
        lin_vel_y_range=(-1.0, 1.0),
        heading_step=0.05,
    ):
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.heading_step = heading_step

        self._lock = threading.Lock()
        self._pressed = set()
        self._heading = 0.0
        self._action = False
        self._reset = False
        self._quit = False
        self._listener = None

    # ---- pynput callbacks (runs in listener thread) ----

    def _on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        with self._lock:
            self._pressed.add(k)
            if k == " ":
                self._action = True
            elif k == "r":
                self._reset = True

    def _on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        with self._lock:
            self._pressed.discard(k)
        # quit on Esc
        if k == "Key.esc":
            with self._lock:
                self._quit = True
            return False  # stop listener

    # ---- public API ----

    def start(self):
        if self._listener is not None:
            return
        try:
            from pynput.keyboard import Key, Listener
        except ImportError:
            raise ImportError(
                "pynput is required for keyboard control. Install with: pip install pynput"
            )
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_commands(self):
        """
        Read current keyboard state. Thread-safe, call from main loop.

        Returns KeyboardCommand namedtuple:
            lin_vel_x, lin_vel_y, heading_target, action, reset, quit
        """
        with self._lock:
            keys = self._pressed.copy()
            action = self._action
            self._action = False
            reset = self._reset
            self._reset = False
            quit_flag = self._quit

        # linear velocity
        lin_vel_x = 0.0
        lin_vel_y = 0.0

        if "w" in keys:
            lin_vel_x = self.lin_vel_x_range[1]
        elif "s" in keys:
            lin_vel_x = self.lin_vel_x_range[0]

        if "d" in keys:
            lin_vel_y = self.lin_vel_y_range[0]
        elif "a" in keys:
            lin_vel_y = self.lin_vel_y_range[1]

        # heading
        if "q" in keys:
            self._heading += self.heading_step
        elif "e" in keys:
            self._heading -= self.heading_step

        import numpy as np

        # wrap heading to [-pi, pi]
        self._heading = np.arctan2(
            np.sin(self._heading), np.cos(self._heading)
        )

        return KeyboardCommand(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            heading_target=self._heading,
            action=action,
            reset=reset,
            quit=quit_flag,
        )
