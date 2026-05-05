"""
consumers.py — RTSP → WebSocket proxy with live face recognition.

Pipeline per frame:
  ffmpeg stdout (raw JPEG bytes)
    → SOI/EOI frame splitter
    → OpenCV decode
    → InsightFace detection (throttled worker thread, RECOG_FPS cap)
    → draw bounding boxes + labels from cached results
    → re-encode to JPEG
    → send as binary WebSocket frame

Status messages (camera connect / retry) are sent as JSON text frames so
the browser can distinguish them from binary video frames.

Retry behaviour: exponential back-off WITHOUT closing the WebSocket, so
the browser does not hammer the camera with rapid reconnects.

Client connects at:
    ws://<host>/ws/stream/
    ws://<host>/ws/stream/?rtsp_url=rtsp://user:pass@host/path
"""

import io
import json
import queue
import subprocess
import threading
import time
import sys
import os
import wave

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from channels.generic.websocket import WebsocketConsumer
import django.db

from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

from .models import FaceMap

# ── stream / recognition config ───────────────────────────────────────────────
DEFAULT_RTSP_URL = "rtsp://admin:trace321@192.168.1.64:554/Streaming/Channels/101"
FPS              = 24           # ffmpeg output frame rate
SCALE            = "1280:-1"    # higher res → better face detail for recognition
                                 # use "640:-1" if CPU can't keep up
RECOG_FPS        = 6            # max face-recognition passes per second
MATCH_THRESH     = 0.4          # cosine distance threshold (lower = stricter)
JPEG_QUALITY     = 95           # re-encode quality (80 introduced visible artefacts
                                 # that degrade recognition; 95 is near-lossless)

# InsightFace detection input size — larger = better accuracy on distant/small faces.
# 640x640 is the default; 1280x1280 is slower but noticeably better at range.
DET_SIZE         = (640, 640)

# ── audio / transcription config ──────────────────────────────────────────────
AUDIO_SAMPLE_RATE  = 16000      # Hz — ElevenLabs expects 16 kHz mono PCM
AUDIO_CHUNK_SEC    = 20         # seconds of audio per ElevenLabs inference call
                                 # shorter = lower latency, higher CPU cost
                                 # longer = more context, better accuracy

ELEVEN_MODEL    = "scribe_v2"
ELEVEN_LANGUAGE = "eng"         # or None for auto-detect

# Silence gate — int16 RMS (0–32768 scale).
# Equivalent to ~0.0003 float RMS; keeps the same sensitivity as the original.
_SILENCE_RMS_INT16 = 10

# Exponential backoff caps (seconds) for ElevenLabs 429 rate-limit responses
_ELEVEN_BACKOFF = [10, 20, 40, 60, 120]

# Retry delays in seconds for RTSP reconnects (mirrors the Node.js RETRY_DELAYS_MS array)
RETRY_DELAYS = [3, 6, 12, 24, 48]

# JPEG byte markers
_SOI = bytes([0xFF, 0xD8])  # start of image
_EOI = bytes([0xFF, 0xD9])  # end of image

# IoU threshold for considering two boxes the "same" face across frames
_IOU_THRESH = 0.35


# ── module-level InsightFace singleton ────────────────────────────────────────
_face_app      = None
_face_app_lock = threading.Lock()


def _get_face_app():
    """Return the shared InsightFace FaceAnalysis instance (created once)."""
    global _face_app
    if _face_app is None:
        with _face_app_lock:
            if _face_app is None:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=0, det_size=DET_SIZE)
                _face_app = app
    return _face_app


# ── module-level ElevenLabs singleton ────────────────────────────────────────
_eleven_client = None
_eleven_lock   = threading.Lock()


def _get_eleven():
    """Shared ElevenLabs client singleton (created once, thread-safe)."""
    global _eleven_client
    if _eleven_client is None:
        with _eleven_lock:
            if _eleven_client is None:
                print("[eleven] Initialising client …")
                _eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                print("[eleven] Client ready.")
    return _eleven_client


# ── module-level embedding cache (shared across all connections) ───────────────
_emb_cache      = []            # list of (name: str, embedding: np.ndarray)
_emb_cache_lock = threading.Lock()
_emb_cache_ts   = 0.0
_EMB_CACHE_TTL  = 5.0           # seconds between DB refreshes


def _get_embeddings() -> list:
    """
    Return a cached list of (name, embedding) tuples.
    Refreshed from the DB at most once every _EMB_CACHE_TTL seconds so
    face-recognition passes never hit the database directly.
    """
    global _emb_cache, _emb_cache_ts
    now = time.monotonic()
    with _emb_cache_lock:
        if now - _emb_cache_ts > _EMB_CACHE_TTL:
            try:
                # Django does not share DB connections across threads.
                # close_old_connections() forces Django to open a fresh
                # connection for this thread instead of reusing a stale one.
                django.db.close_old_connections()
                _emb_cache = [
                    (f.name, np.array(f.array, dtype=np.float32))
                    for f in FaceMap.objects.all()
                    if f.array
                ]
                _emb_cache_ts = now
                print(f"[embeddings] Cache refreshed — {len(_emb_cache)} face(s) loaded")
            except Exception as exc:
                print(f"[embeddings] DB refresh error: {exc}")
                # Keep serving the previous cache rather than crashing
        return list(_emb_cache)


# ── audio helpers ─────────────────────────────────────────────────────────────

def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = AUDIO_SAMPLE_RATE) -> bytes:
    """
    Wrap raw 16-bit mono PCM in a WAV container entirely in memory.
    Avoids all disk I/O that a tempfile approach would incur.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)           # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _is_silent(pcm: bytes, threshold: int = _SILENCE_RMS_INT16) -> bool:
    """
    Fast silence gate on raw int16 PCM.
    Operates on the native int16 dtype — no float conversion needed —
    which is ~2× faster and avoids a temporary array allocation.
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    return int(np.abs(samples).mean()) < threshold


# ── drawing helpers ───────────────────────────────────────────────────────────

def _draw_face(frame: np.ndarray, bbox, name: str, dist: float, matched: bool):
    """Draw a coloured bounding box and label on frame (in-place)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    color = (0, 220, 0) if matched else (0, 0, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{name}  d={dist:.2f}" if dist < 999.0 else name
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


def _draw_fps(frame: np.ndarray, fps: float):
    """Stamp FPS counter in the top-left corner (in-place)."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )


# ── face recognition helper ───────────────────────────────────────────────────

def _match_face(embedding: np.ndarray, embeddings: list) -> tuple:
    """
    Compare embedding against the cached DB list.
    Mirrors the Recognize viewset algorithm exactly:
      - Track best_name and best_distance independently
      - best_match_found is set only when distance < threshold
      - If no match found, name falls back to "Unknown" at the end
      - best_distance is always the true closest distance regardless of threshold
    Returns (name, distance, match_found).
    """
    best_name        = "Unknown"
    best_distance    = 999.0
    best_match_found = False

    for name, stored_emb in embeddings:
        dist = cosine(embedding, stored_emb)
        if dist < best_distance:
            best_distance    = dist
            best_name        = name
            best_match_found = dist < MATCH_THRESH

    # If the closest candidate didn't beat the threshold, report Unknown
    if not best_match_found:
        best_name = "Unknown"

    return best_name, best_distance, best_match_found


# ── persistence tracker helpers ──────────────────────────────────────────────

def _iou(a, b) -> float:
    """
    Intersection-over-Union between two bounding boxes [x1,y1,x2,y2].
    Used to match a current detection to a previously seen face slot.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ── consumer ──────────────────────────────────────────────────────────────────

class RTSPProxyConsumer(WebsocketConsumer):
    """
    Synchronous Channels consumer: RTSP stream → face recognition → WebSocket.

    Thread layout
    ─────────────
    Channels worker thread   — runs connect / disconnect / receive
    _read_stdout thread      — reads ffmpeg stdout, splits JPEG frames,
                               submits to recognition queue, annotates +
                               sends frames back over the WebSocket
    _recognition_worker      — pulls frames from _recog_queue, runs
                               InsightFace, stores results in _recog_results
    _drain_stderr thread     — forwards ffmpeg stderr to server stderr
    _watch_exit thread       — waits for ffmpeg to exit and schedules retry
    _read_audio_stdout thread — reads raw PCM from audio ffmpeg, accumulates
                               chunks and enqueues for ElevenLabs
    _eleven_worker thread    — pulls PCM chunks from queue, transcribes via
                               ElevenLabs Scribe, sends transcript frames
    """

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self):
        self.accept()

        # Parse rtsp_url from query string, fall back to default
        qs       = self.scope.get("query_string", b"").decode()
        params   = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        rtsp_url = params.get("rtsp_url", DEFAULT_RTSP_URL)

        self._rtsp_url    = rtsp_url
        self._closed      = False
        self._attempt     = 0
        self._active_proc = None
        self._retry_timer = None

        # Frame buffer for ffmpeg stdout parsing
        self._buf      = bytearray()
        self._buf_lock = threading.Lock()

        # Recognition worker state.
        # Single-slot queue: always holds the *latest* frame to recognise.
        # If the worker is busy the stale queued frame is replaced so we
        # never fall behind — same approach as the standalone script's
        # RECOG_FPS throttle.
        self._recog_queue    = queue.Queue(maxsize=1)
        self._recog_results  = []       # list of (bbox, name, dist, matched)
        self._recog_lock     = threading.Lock()
        self._last_recog_ts  = 0.0
        self._recog_interval = 1.0 / RECOG_FPS

        # Persistence tracker.
        # Maps a slot index → {"bbox": ..., "name": ..., "dist": ..., "matched": bool}
        # Once a face slot becomes matched (green), it stays green until that
        # face is no longer detected (IoU with any current detection drops to 0).
        self._persist        = {}       # {slot_id: dict}
        self._persist_lock   = threading.Lock()
        self._persist_slot   = 0        # ever-incrementing slot id counter

        # Names already announced during this websocket session.
        # Prevents duplicate recognition events even if a person leaves
        # frame and re-enters later.
        self._recognized_names = set()

        # Audio transcription state.
        # A second ffmpeg process pulls raw PCM audio from the same RTSP URL.
        # The chunker thread accumulates samples and fires ElevenLabs every
        # AUDIO_CHUNK_SEC seconds via a queue, mirroring the recognition
        # worker pattern.
        self._audio_proc      = None    # ffmpeg audio process
        self._audio_queue     = queue.Queue(maxsize=4)
        self._pcm_buf         = bytearray()
        self._pcm_lock        = threading.Lock()
        self._bytes_per_chunk = AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SEC * 2  # 16-bit mono

        # ElevenLabs circuit breaker + backoff state
        self._eleven_disabled_until = 0.0
        self._eleven_backoff_idx    = 0

        # FPS tracking
        self._fps_counter = 0
        self._fps_display = 0.0
        self._fps_ts      = time.monotonic()
        self._fps_lock    = threading.Lock()

        # Start recognition worker before ffmpeg so it's ready immediately
        threading.Thread(target=self._recognition_worker, daemon=True).start()

        print(f"[rtsp-proxy] Client connected → {rtsp_url}")
        self._launch_ffmpeg()
        # _eleven_worker is started inside _launch_audio_ffmpeg() so it is
        # tied to the audio process actually being alive.
        self._launch_audio_ffmpeg()

    def disconnect(self, code):
        self._closed = True
        print("[rtsp-proxy] Client disconnected")

        # Stop retry timer
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

        # Kill video ffmpeg
        proc = self._active_proc
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            self._active_proc = None

        # Send sentinel to unblock recognition worker so its thread exits
        try:
            self._recog_queue.put_nowait(None)
        except Exception:
            pass

        # Kill audio ffmpeg process
        audio_proc = self._audio_proc
        if audio_proc is not None:
            try:
                audio_proc.kill()
            except OSError:
                pass
            self._audio_proc = None

        # Send sentinel to unblock ElevenLabs worker
        try:
            self._audio_queue.put_nowait(None)
        except Exception:
            pass

    def receive(self, text_data=None, bytes_data=None):
        pass  # browser never sends data

    # ── recognition worker ────────────────────────────────────────────────────

    def _recognition_worker(self):
        """
        Pull frames from _recog_queue, run InsightFace, cache results.
        Runs in its own daemon thread so heavy ML work never blocks sending.
        """
        app = _get_face_app()

        while True:
            frame = self._recog_queue.get()   # blocks until a frame arrives
            if frame is None:
                break                          # sentinel — shut down cleanly

            try:
                embeddings     = _get_embeddings()
                faces          = app.get(frame)
                current_bboxes = [face.bbox for face in faces]

                # ── Step 1: Expire slots whose face is no longer detected ──────
                # A slot is expired when no current detection overlaps it
                # above IOU_THRESH — meaning that face left the frame.
                with self._persist_lock:
                    expired = [
                        sid for sid, slot in self._persist.items()
                        if not any(
                            _iou(slot["bbox"], bbox) >= _IOU_THRESH
                            for bbox in current_bboxes
                        )
                    ]
                    for sid in expired:
                        del self._persist[sid]

                # ── Step 2: Run recognition on each detected face ─────────────
                results = []
                for face in faces:
                    bbox                = face.bbox
                    emb                 = face.embedding.astype(np.float32)
                    name, dist, matched = _match_face(emb, embeddings)

                    with self._persist_lock:
                        # Find the existing slot for this face (by IoU overlap)
                        matched_slot = None
                        for sid, slot in self._persist.items():
                            if _iou(slot["bbox"], bbox) >= _IOU_THRESH:
                                matched_slot = sid
                                break

                        if matched_slot is not None:
                            slot = self._persist[matched_slot]
                            # Always update bbox to follow the face as it moves
                            slot["bbox"] = bbox
                            # Only upgrade: red → green (never downgrade green → red)
                            if matched and not slot["matched"]:
                                slot["name"]    = name
                                slot["dist"]    = dist
                                slot["matched"] = True

                                # SESSION-level first recognition only
                                if name != "Unknown" and name not in self._recognized_names:
                                    iso = self._iso_now()
                                    slot["recognized_at"] = iso
                                    self._recognized_names.add(name)
                                    self._send_recognition(name=name, recognized_at=iso, distance=dist)

                            elif slot["matched"]:
                                # Already green — keep the confirmed identity,
                                # update distance if we get a better read
                                if matched and dist < slot["dist"]:
                                    slot["name"] = name
                                    slot["dist"] = dist
                        else:
                            # New face — create a fresh slot
                            sid = self._persist_slot
                            self._persist_slot += 1
                            self._persist[sid] = {
                                "bbox":          bbox,
                                "name":          name,
                                "dist":          dist,
                                "matched":       matched,
                                "recognized_at": None,
                            }

                            # If a brand-new face is already matched on first sight,
                            # emit session-level first recognition event.
                            if matched and name != "Unknown" and name not in self._recognized_names:
                                iso = self._iso_now()
                                self._persist[sid]["recognized_at"] = iso
                                self._recognized_names.add(name)
                                self._send_recognition(name=name, recognized_at=iso, distance=dist)

                    # Use the persisted state for drawing
                    with self._persist_lock:
                        for slot in self._persist.values():
                            if _iou(slot["bbox"], bbox) >= _IOU_THRESH:
                                results.append((
                                    bbox,
                                    slot["name"],
                                    slot["dist"],
                                    slot["matched"],
                                ))
                                break

                # Sort so largest face is drawn last (on top)
                results.sort(
                    key=lambda r: (r[0][2] - r[0][0]) * (r[0][3] - r[0][1])
                )

                print(f"[recognition] {len(faces)} face(s) detected, {len(embeddings)} embedding(s) in cache")

                with self._recog_lock:
                    self._recog_results = results

            except Exception as exc:
                import traceback
                print(f"[recognition] Error: {exc}")
                traceback.print_exc()

    def _submit_for_recognition(self, frame: np.ndarray):
        """
        Rate-limited submission to the recognition worker.
        Replaces any stale queued frame with the latest one (non-blocking).
        """
        now = time.monotonic()
        if now - self._last_recog_ts < self._recog_interval:
            return
        self._last_recog_ts = now

        try:
            self._recog_queue.get_nowait()   # evict stale frame if present
        except queue.Empty:
            pass
        try:
            self._recog_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    # ── frame annotation ──────────────────────────────────────────────────────

    def _annotate_and_send(self, jpeg_bytes: bytes):
        """
        Decode JPEG → annotate with cached recognition results + FPS →
        re-encode to JPEG → send as binary WebSocket frame.
        """
        arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Submit to recognition worker (rate-limited, non-blocking)
        self._submit_for_recognition(frame)

        # Stamp the latest cached detection results onto the display frame
        with self._recog_lock:
            results = list(self._recog_results)

        for bbox, name, dist, matched in results:
            _draw_face(frame, bbox, name, dist, matched)

        # Update and draw FPS counter
        with self._fps_lock:
            self._fps_counter += 1
            now     = time.monotonic()
            elapsed = now - self._fps_ts
            if elapsed >= 1.0:
                self._fps_display = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_ts      = now
            fps = self._fps_display

        # _draw_fps(frame, fps)

        # Re-encode and send
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            self._send_bytes(buf.tobytes())

    # ── ffmpeg management ─────────────────────────────────────────────────────

    def _launch_ffmpeg(self):
        """Spawn ffmpeg and start stdout / stderr / exit-watcher threads."""
        if self._closed:
            return

        label = "Connecting…" if self._attempt == 0 \
                else f"Reconnecting (attempt {self._attempt + 1})…"
        print(f"[rtsp-proxy] {label}")
        self._send_status(label)

        cmd = [
            "ffmpeg",
            "-loglevel",       "error",
            "-rtsp_transport", "tcp",
            "-i",              self._rtsp_url,
            "-f",              "image2pipe",
            "-vcodec",         "mjpeg",
            "-vf",             f"fps={FPS},scale={SCALE}",
            "-q:v",            "1",    # highest MJPEG quality from ffmpeg
            "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            print("[rtsp-proxy] ffmpeg not found — is it installed and on PATH?")
            self._send_status("ffmpeg not found on server")
            return
        except OSError as exc:
            print(f"[rtsp-proxy] spawn failed: {exc}")
            self._schedule_retry()
            return

        self._active_proc = proc
        started_at        = time.monotonic()

        threading.Thread(target=self._read_stdout,  args=(proc,),            daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(proc,),            daemon=True).start()
        threading.Thread(target=self._watch_exit,   args=(proc, started_at), daemon=True).start()

    def _read_stdout(self, proc: subprocess.Popen):
        """
        Read raw bytes from ffmpeg stdout, scan for JPEG SOI/EOI markers,
        and pass each complete frame through _annotate_and_send.
        """
        CHUNK = 65536
        try:
            while not self._closed:
                chunk = proc.stdout.read(CHUNK)
                if not chunk:
                    break

                # Collect complete frames under the lock, annotate OUTSIDE
                # the lock — annotation is expensive and must not block the buffer.
                frames_to_send = []
                with self._buf_lock:
                    self._buf.extend(chunk)
                    while True:
                        start = self._buf.find(_SOI)
                        if start == -1:
                            self._buf.clear()
                            break
                        end = self._buf.find(_EOI, start + 2)
                        if end == -1:
                            if start > 0:
                                del self._buf[:start]
                            break
                        end += 2
                        frames_to_send.append(bytes(self._buf[start:end]))
                        del self._buf[:end]

                for jpeg in frames_to_send:
                    self._annotate_and_send(jpeg)

        except OSError:
            pass

    def _drain_stderr(self, proc: subprocess.Popen):
        """Forward ffmpeg stderr to server stderr for debugging."""
        try:
            for line in proc.stderr:
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()
        except OSError:
            pass

    def _watch_exit(self, proc: subprocess.Popen, started_at: float):
        """Wait for ffmpeg to exit; schedule retry with exponential back-off."""
        proc.wait()

        if self._closed:
            return
        if self._active_proc is not proc:
            return  # a newer process has already replaced this one

        self._active_proc = None
        uptime = time.monotonic() - started_at

        if uptime < 8.0:
            # Quick exit → camera connection failure → back-off retry
            self._schedule_retry()
        else:
            # Was streaming, then dropped → reset counter and retry quickly
            self._attempt = 0
            self._schedule_retry()

    # ── retry / back-off ──────────────────────────────────────────────────────

    def _schedule_retry(self):
        if self._closed:
            return

        if self._attempt >= len(RETRY_DELAYS):
            print("[rtsp-proxy] Max retries reached — closing connection")
            self._send_status("Camera unreachable after multiple attempts")
            self.close()
            return

        delay = RETRY_DELAYS[self._attempt]
        print(f"[rtsp-proxy] Retry {self._attempt + 1} in {delay}s…")
        self._send_status(f"Camera busy — retrying in {delay}s…")

        self._retry_timer = threading.Timer(delay, self._do_retry)
        self._retry_timer.daemon = True
        self._retry_timer.start()

    def _do_retry(self):
        self._retry_timer = None
        self._attempt    += 1
        with self._buf_lock:
            self._buf.clear()
        self._launch_ffmpeg()

    # ── audio capture ─────────────────────────────────────────────────────────

    def _launch_audio_ffmpeg(self):
        """
        Spawn a second ffmpeg process that pulls ONLY audio from the RTSP
        stream and pipes raw 16-bit mono PCM at 16 kHz to stdout.
        Runs completely independently of the video ffmpeg process.
        The ElevenLabs worker thread is started here so it is tied to the
        audio process actually being alive.
        """
        if self._closed:
            return

        cmd = [
            "ffmpeg",
            "-loglevel",       "error",
            "-rtsp_transport", "tcp",
            "-i",              self._rtsp_url,
            "-vn",                           # no video
            "-acodec",         "pcm_s16le",  # raw 16-bit signed little-endian PCM
            "-ar",             str(AUDIO_SAMPLE_RATE),
            "-ac",             "1",          # mono
            "-f",              "s16le",      # raw PCM container
            "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,   # suppress audio ffmpeg noise
                bufsize=0,
            )
        except (FileNotFoundError, OSError) as exc:
            print(f"[audio] ffmpeg spawn failed: {exc}")
            return

        self._audio_proc = proc
        print("[audio] ffmpeg started — capturing audio stream")

        threading.Thread(target=self._read_audio_stdout, args=(proc,), daemon=True).start()
        # Start ElevenLabs worker only after audio ffmpeg is confirmed alive
        threading.Thread(target=self._eleven_worker, daemon=True).start()

    def _read_audio_stdout(self, proc: subprocess.Popen):
        """
        Read raw PCM bytes from audio ffmpeg stdout.
        Accumulates samples in _pcm_buf and enqueues a chunk to the
        ElevenLabs worker every AUDIO_CHUNK_SEC seconds.

        Backpressure: if the queue is full (ElevenLabs is falling behind),
        evict the oldest (stale) chunk and insert the freshest one so the
        worker always processes the most recent audio.
        """
        CHUNK = 4096
        try:
            while not self._closed:
                chunk = proc.stdout.read(CHUNK)
                if not chunk:
                    break

                with self._pcm_lock:
                    self._pcm_buf.extend(chunk)
                    while len(self._pcm_buf) >= self._bytes_per_chunk:
                        pcm_chunk = bytes(self._pcm_buf[:self._bytes_per_chunk])
                        del self._pcm_buf[:self._bytes_per_chunk]

                        # Non-blocking enqueue: if full, evict the oldest
                        # stale chunk and insert the freshest one instead.
                        if self._audio_queue.full():
                            try:
                                self._audio_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self._audio_queue.put_nowait(pcm_chunk)

        except OSError:
            pass

    # ── ElevenLabs transcription worker ───────────────────────────────────────

    def _eleven_worker(self):
        """
        Pull PCM chunks from _audio_queue and transcribe via ElevenLabs Scribe.

        Key behaviours:
        - In-memory WAV wrapping (no temp files, zero disk I/O)
        - Fast int16 silence gate (no float conversion)
        - Per-speaker transcript segments when diarization is available
        - Exponential backoff on 429 rate-limit responses (non-blocking)
        - Circuit breaker for 401 (unusual activity) with 30-minute blackout;
          queue is drained during the blackout so stale audio is discarded
        """
        client = _get_eleven()

        while True:
            # ── Circuit breaker ───────────────────────────────────────────────
            now = time.time()
            if now < self._eleven_disabled_until:
                # Sleep in short increments so the disconnect sentinel is seen promptly
                time.sleep(min(2.0, self._eleven_disabled_until - now))
                # Drain backlog that built up during the blackout
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
                continue

            pcm_chunk = self._audio_queue.get()   # blocks until chunk or sentinel
            if pcm_chunk is None:
                break                               # disconnect sentinel

            # ── Silence gate (fast int16 path) ────────────────────────────────
            if _is_silent(pcm_chunk):
                print("[eleven] skipped silence")
                self._eleven_backoff_idx = 0        # reset backoff on clean path
                continue

            try:
                wav_bytes = _pcm_to_wav_bytes(pcm_chunk)

                result = client.speech_to_text.convert(
                    file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                    model_id=ELEVEN_MODEL,
                    tag_audio_events=True,
                    language_code=ELEVEN_LANGUAGE,
                    diarize=True,
                )

                text = self._extract_transcript(result)
                print(f"[eleven] result: {text!r}")

                if text:
                    self._send_transcript(text)

                self._eleven_backoff_idx = 0        # successful call — reset backoff

            except ApiError as exc:
                status = getattr(exc, "status_code", None)

                if status == 401:
                    print(
                        "[eleven] 401 — unusual activity detected. "
                        "Pausing transcription for 30 minutes."
                    )
                    self._eleven_disabled_until = time.time() + 1800
                    continue

                if status == 429:
                    delay = _ELEVEN_BACKOFF[
                        min(self._eleven_backoff_idx, len(_ELEVEN_BACKOFF) - 1)
                    ]
                    self._eleven_backoff_idx = min(
                        self._eleven_backoff_idx + 1, len(_ELEVEN_BACKOFF) - 1
                    )
                    print(f"[eleven] Rate limited — backing off {delay}s")
                    time.sleep(delay)
                    continue

                print(f"[eleven] API error {status}: {exc}")

            except Exception as exc:
                import traceback
                print(f"[eleven] Unexpected error: {exc}")
                traceback.print_exc()

    @staticmethod
    def _extract_transcript(result) -> str:
        """
        Pull the best available text from an ElevenLabs Scribe response.

        Priority:
          1. Diarized utterances  → "Speaker 0: hello  Speaker 1: world"
          2. Flat .text / dict["text"]
          3. Empty string (caller skips sending)
        """
        # Diarized path — list of utterance objects with .speaker + .text
        utterances = None
        if hasattr(result, "utterances"):
            utterances = result.utterances
        elif isinstance(result, dict):
            utterances = result.get("utterances")

        if utterances:
            parts = []
            for u in utterances:
                speaker = getattr(u, "speaker", None) or u.get("speaker", "?")
                utext   = (getattr(u, "text", None) or u.get("text", "")).strip()
                if utext:
                    parts.append(f"Speaker {speaker}: {utext}")
            if parts:
                return "  ".join(parts)

        # Flat text fallback
        if hasattr(result, "text"):
            return result.text.strip()
        if isinstance(result, dict):
            return result.get("text", "").strip()

        return ""

    # ── shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _iso_now() -> str:
        """Return the current UTC time as a millisecond-precision ISO 8601 string."""
        ts = time.time()
        return (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))
            + f".{int((ts % 1) * 1000):03d}Z"
        )

    # ── thread-safe WebSocket send helpers ────────────────────────────────────

    def _send_bytes(self, data: bytes):
        """Send annotated JPEG frame as a binary WebSocket message."""
        try:
            self.send(bytes_data=data)
        except Exception:
            pass

    def _send_status(self, msg: str):
        """Send a JSON status/control text frame."""
        try:
            self.send(text_data=json.dumps({"type": "status", "msg": msg}))
        except Exception:
            pass

    def _send_transcript(self, text: str):
        """Send an ElevenLabs transcript as a JSON text frame."""
        try:
            self.send(text_data=json.dumps({"type": "transcript", "text": text}))
        except Exception:
            pass

    def _send_recognition(self, name: str, recognized_at: str, distance: float):
        """
        Send first-ever recognition event for this WebSocket session.
        One event per unique person name.
        """
        try:
            self.send(
                text_data=json.dumps({
                    "type":           "recognition",
                    "name":           name,
                    "recognized_at":  recognized_at,
                    "distance":       round(float(distance), 4),
                })
            )
        except Exception:
            pass