"""Multi-camera safety watchdog dashboard (hackathon demo).

Plays N camera videos in a dark, styled grid, runs the watchdog (YOLO/YOLOE
detection + knife orientation + danger rules) on each feed, and streams a live
event log. Inference runs in a background thread so the video stays smooth on
CPU while detections/danger update asynchronously.

    uv run python dashboard.py                 # auto-uses the 3 default videos
    uv run python dashboard.py a.mp4 b.mp4 c.mp4

Press q (or Esc) to quit.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

import config as cfg
from src.detector import Detector
from src.orientation import blade_geometry
from src.rules import RuleEngine

# ---- default feeds (label inferred from filename) --------------------------
DEFAULT_VIDEOS = ["file-000.mp4", "file-000-gripper.mp4", "WIN_20260627_21_00_58_Pro.mp4"]

# ---- palette (BGR) ---------------------------------------------------------
BG = (18, 18, 22)
PANEL = (30, 30, 36)
EDGE = (60, 60, 70)
TEXT = (235, 235, 235)
MUTED = (140, 140, 150)
GREEN = (90, 210, 110)
AMBER = (40, 180, 255)
RED = (60, 60, 240)
ACCENT = (255, 170, 50)
SEV_COLOR = {"critical": RED, "warning": AMBER, None: GREEN, "clear": GREEN}

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_DUPLEX


def label_for(path: str) -> str:
    p = path.lower()
    if "gripper" in p:
        return "GRIPPER CAM"
    if "win_" in p or "scene" in p:
        return "SCENE CAM"
    return "HEAD CAM"


class Feed:
    def __init__(self, path: str):
        self.path = path
        self.label = label_for(path)
        self.cap = cv2.VideoCapture(path)
        self.frame = None
        self.dets = []
        self.severity = None
        self.lock = threading.Lock()

    def read_next(self):
        ok, f = self.cap.read()
        if not ok:  # loop the video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if ok:
            with self.lock:
                self.frame = f
        return ok


class Dashboard:
    def __init__(self, videos):
        self.feeds = [Feed(v) for v in videos]
        self.detector = Detector.build(cfg.CONFIG)
        self.rules = [RuleEngine(cfg.CONFIG.thresholds) for _ in self.feeds]
        self.logs: deque[tuple[float, str, str, str]] = deque(maxlen=200)
        self.log_lock = threading.Lock()
        self._stop = False
        self._t0 = None  # set on first frame (no Date.now in some envs is fine here)

    # ---- background inference (round-robin over cams) ----------------------
    def _infer_loop(self):
        i = 0
        while not self._stop:
            feed = self.feeds[i % len(self.feeds)]
            i += 1
            with feed.lock:
                frame = None if feed.frame is None else feed.frame.copy()
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                dets = self.detector.detect(frame)
            except Exception:
                continue
            analysis = self.rules[self.feeds.index(feed)].analyze(dets, [])
            with feed.lock:
                feed.dets = dets
                feed.severity = analysis.max_severity
            for h in analysis.hits:
                self._log(feed.label, h.severity, h.reason)

    def _log(self, cam, severity, msg):
        with self.log_lock:
            # de-dup consecutive identical messages from the same cam
            if self.logs and self.logs[-1][1] == cam and self.logs[-1][3] == msg:
                return
            self.logs.append((time.time(), cam, severity, msg))

    # ---- rendering ---------------------------------------------------------
    def run(self):
        worker = threading.Thread(target=self._infer_loop, daemon=True)
        worker.start()
        self._log("SYSTEM", "warning", "Watchdog online — monitoring 3 feeds")
        win = "Robot Safety Watchdog"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 900)
        try:
            while True:
                for f in self.feeds:
                    f.read_next()
                canvas = self._compose()
                cv2.imshow(win, canvas)
                k = cv2.waitKey(15) & 0xFF
                if k in (ord("q"), 27):
                    break
        finally:
            self._stop = True
            for f in self.feeds:
                f.cap.release()
            cv2.destroyAllWindows()

    def _compose(self) -> np.ndarray:
        W, H = 1600, 900
        c = np.full((H, W, 3), BG, np.uint8)
        self._header(c, W)

        # camera row
        n = len(self.feeds)
        margin, top, gap = 20, 70, 16
        tiles_h = 470
        tw = (W - 2 * margin - (n - 1) * gap) // n
        worst = None
        for idx, feed in enumerate(self.feeds):
            x = margin + idx * (tw + gap)
            with feed.lock:
                frame = None if feed.frame is None else feed.frame.copy()
                dets = list(feed.dets)
                sev = feed.severity
            self._tile(c, x, top, tw, tiles_h, feed.label, frame, dets, sev)
            if sev == "critical" or (sev == "warning" and worst != "critical"):
                worst = sev
        self._status_pill(c, W, worst)

        # log panel
        self._logs_panel(c, margin, top + tiles_h + 16, W - 2 * margin,
                         H - (top + tiles_h + 16) - margin)
        return c

    def _header(self, c, W):
        cv2.rectangle(c, (0, 0), (W, 56), PANEL, -1)
        cv2.line(c, (0, 56), (W, 56), EDGE, 1)
        cv2.circle(c, (24, 28), 6, RED, -1)  # REC dot
        cv2.putText(c, "ROBOT SAFETY WATCHDOG", (42, 36), MONO, 0.8, TEXT, 1, cv2.LINE_AA)
        cv2.putText(c, "multi-cam  -  empirical safety monitor", (340, 35), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
        clock = time.strftime("%H:%M:%S")
        cv2.putText(c, clock, (W - 120, 36), MONO, 0.7, TEXT, 1, cv2.LINE_AA)

    def _status_pill(self, c, W, worst):
        col = SEV_COLOR.get(worst, GREEN)
        txt = {"critical": "CRITICAL", "warning": "WARNING"}.get(worst, "ALL CLEAR")
        (tw_, th_), _ = cv2.getTextSize(txt, MONO, 0.7, 1)
        x2, y = W - 150 - 170, 36
        cv2.rectangle(c, (x2 - 14, 12), (x2 + tw_ + 14, 44), col, -1)
        cv2.putText(c, txt, (x2, y), MONO, 0.7, (15, 15, 18), 1, cv2.LINE_AA)

    def _tile(self, c, x, y, w, h, label, frame, dets, sev):
        col = SEV_COLOR.get(sev, GREEN)
        cv2.rectangle(c, (x, y), (x + w, y + h), PANEL, -1)
        # video area
        vy0, vy1 = y + 34, y + h - 10
        vw, vh = w - 20, vy1 - vy0
        if frame is not None:
            annotated = self._annotate(frame, dets)
            lb = self._letterbox(annotated, vw, vh)
            c[vy0:vy0 + vh, x + 10:x + 10 + vw] = lb
        else:
            cv2.putText(c, "connecting...", (x + 20, vy0 + 30), FONT, 0.6, MUTED, 1)
        # label bar + danger badge
        cv2.putText(c, label, (x + 12, y + 24), MONO, 0.6, TEXT, 1, cv2.LINE_AA)
        badge = {"critical": "DANGER", "warning": "WATCH"}.get(sev, "OK")
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.5, 1)
        cv2.rectangle(c, (x + w - bw - 26, y + 8), (x + w - 6, y + 28), col, -1)
        cv2.putText(c, badge, (x + w - bw - 18, y + 23), FONT, 0.5, (15, 15, 18), 1, cv2.LINE_AA)
        cv2.rectangle(c, (x, y), (x + w, y + h), col if sev else EDGE, 2)

    @staticmethod
    def _letterbox(img, w, h):
        ih, iw = img.shape[:2]
        s = min(w / iw, h / ih)
        nw, nh = int(iw * s), int(ih * s)
        r = cv2.resize(img, (nw, nh))
        out = np.full((h, w, 3), (12, 12, 15), np.uint8)
        ox, oy = (w - nw) // 2, (h - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = r
        return out

    def _annotate(self, frame, dets):
        f = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            col = RED if d.label in cfg.SHARP_CLASSES else (200, 200, 205)
            cv2.rectangle(f, (x1, y1), (x2, y2), col, 2)
            cv2.putText(f, f"{d.label} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                        FONT, 0.5, col, 1, cv2.LINE_AA)
            if d.label in cfg.SHARP_CLASSES and d.mask is not None:
                geo = blade_geometry(d.mask)
                if geo is not None:
                    tip = (int(geo.tip[0]), int(geo.tip[1]))
                    hd = (int(geo.handle[0]), int(geo.handle[1]))
                    cv2.line(f, hd, tip, AMBER, 2)
                    cv2.circle(f, tip, 6, RED, -1)
        return f

    def _logs_panel(self, c, x, y, w, h):
        cv2.rectangle(c, (x, y), (x + w, y + h), PANEL, -1)
        cv2.rectangle(c, (x, y), (x + w, y + h), EDGE, 1)
        cv2.putText(c, "EVENT LOG", (x + 12, y + 22), MONO, 0.55, ACCENT, 1, cv2.LINE_AA)
        with self.log_lock:
            rows = list(self.logs)[-12:][::-1]
        ly = y + 46
        for ts, cam, sev, msg in rows:
            col = SEV_COLOR.get(sev, GREEN)
            t = time.strftime("%H:%M:%S", time.localtime(ts))
            icon = {"critical": "!!", "warning": "! ", None: "  "}.get(sev, "  ")
            line = f"{t}  {icon} [{cam:11s}] {msg}"
            cv2.putText(c, line[:120], (x + 12, ly), MONO, 0.42, col, 1, cv2.LINE_AA)
            ly += 20
            if ly > y + h - 8:
                break


def main():
    videos = sys.argv[1:] or DEFAULT_VIDEOS
    Dashboard(videos).run()


if __name__ == "__main__":
    main()
