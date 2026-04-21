from django.shortcuts import render
from django.http import StreamingHttpResponse
from .models import FaceMap
from .serializers import FaceMapSerializer, FaceImageSerializer, RecognizeSerializer
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import os
import cv2
import threading
import time
import queue

# Initialize InsightFace model (cached at module level)
_face_app = None

def get_face_app():
    """Get or initialize the InsightFace FaceAnalysis app."""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app

# ─────────────────────────────────────────────
# Stream Processing Classes
# ─────────────────────────────────────────────

class RTSPStreamReader(threading.Thread):
    """Read RTSP stream in a background thread with automatic reconnection."""

    def __init__(self, rtsp_url: str, buffer_size: int = 2):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self.connected = False
        # NOTE: start() is called explicitly by StreamProcessor after full init

    def run(self):
        """Background thread: open capture, drain frames, reconnect on failure."""
        while not self._stop_event.is_set():
            print(f"[Stream] Connecting to {self.rtsp_url}")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("[Stream] Failed to connect. Retrying in 3s...")
                time.sleep(3)
                continue

            print("[Stream] Connected!")
            self.connected = True
            consecutive_fails = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_fails += 1
                    if consecutive_fails > 10:
                        print("[Stream] Stream lost. Reconnecting...")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_fails = 0
                # Keep queue at most 1 frame deep — always show latest
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)

            cap.release()
            self.connected = False
            if not self._stop_event.is_set():
                time.sleep(3)

    def get_frame(self):
        """Return the latest frame, or None if none is available yet."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Signal the reader thread to exit."""
        self._stop_event.set()


# ─────────────────────────────────────────────
# Embedding Cache
# ─────────────────────────────────────────────

# Refreshed every N seconds to avoid per-frame DB hits
_embedding_cache: list = []
_embedding_cache_lock = threading.Lock()
_embedding_cache_ts = 0.0
_EMBEDDING_CACHE_TTL = 5.0  # seconds


def _get_embeddings() -> list:
    """
    Return a cached list of (name, embedding) tuples.
    The cache is refreshed at most once every _EMBEDDING_CACHE_TTL seconds,
    so the DB is not queried on every single frame.
    """
    global _embedding_cache, _embedding_cache_ts
    now = time.monotonic()
    with _embedding_cache_lock:
        if now - _embedding_cache_ts > _EMBEDDING_CACHE_TTL:
            _embedding_cache = [
                (f.name, np.array(f.array, dtype=np.float32))
                for f in FaceMap.objects.all()
                if f.array
            ]
            _embedding_cache_ts = now
        return list(_embedding_cache)


# ─────────────────────────────────────────────
# Stream Processor
# ─────────────────────────────────────────────

class StreamProcessor:
    """Attach a reader to a specific RTSP URL and annotate frames with face-recognition results."""

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.reader = RTSPStreamReader(rtsp_url)
        self.reader.start()  # start AFTER the object is fully initialised
        self.app = get_face_app()

    def process_frame(self, frame: np.ndarray):
        """Detect faces in frame, draw labelled bounding boxes, return annotated copy."""
        if frame is None:
            return None

        display = frame.copy()
        try:
            faces = self.app.get(frame)
            if not faces:
                return display

            embeddings = _get_embeddings()

            for face in faces:
                bbox      = face.bbox
                embedding = face.embedding.astype(np.float32)

                best_name     = "Unknown"
                best_distance = 999.0

                for name, stored_emb in embeddings:
                    dist = cosine(embedding, stored_emb)
                    if dist < best_distance:
                        best_distance = dist
                        best_name     = name

                matched = best_distance < 0.4

                x1, y1, x2, y2 = (int(v) for v in bbox)
                color = (0, 220, 0) if matched else (0, 0, 220)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                label = f"{best_name} ({best_distance:.2f})"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    display,
                    (x1, y1 - th - baseline - 4),
                    (x1 + tw + 4, y1),
                    color, -1,
                )
                cv2.putText(
                    display, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        except Exception as e:
            print(f"[Processor] Frame processing error: {e}")

        return display

    def stream_generator(self, stop_event: threading.Event):
        """
        Yield Motion-JPEG frames until stop_event is set.
        The stop_event is owned by the view and is set when the HTTP
        connection closes, so the generator — and the reader thread —
        are guaranteed to be cleaned up.
        """
        while not stop_event.is_set():
            frame = self.reader.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            processed = self.process_frame(frame)
            if processed is None:
                continue

            ret, jpeg = cv2.imencode(
                '.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ret:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + jpeg.tobytes()
                + b'\r\n'
            )

    def stop(self):
        """Stop the underlying reader thread."""
        self.reader.stop()


# ─────────────────────────────────────────────
# Per-URL Processor Registry
# ─────────────────────────────────────────────

_stream_processors: dict = {}
_stream_processors_lock = threading.Lock()


def get_or_create_stream_processor(rtsp_url: str) -> StreamProcessor:
    """
    Return the StreamProcessor for rtsp_url, creating one if necessary.
    Each distinct URL gets its own processor; calling the endpoint again
    with the same URL reuses the existing processor rather than spawning
    duplicate reader threads.
    """
    with _stream_processors_lock:
        if rtsp_url not in _stream_processors:
            _stream_processors[rtsp_url] = StreamProcessor(rtsp_url)
        return _stream_processors[rtsp_url]


# ─────────────────────────────────────────────
# ViewSets
# ─────────────────────────────────────────────

class Register(viewsets.ModelViewSet):
    queryset = FaceMap.objects.all()
    serializer_class = FaceMapSerializer
    http_method_names = ['post', 'get', 'put', 'patch', 'delete']

    def create(self, request, *args, **kwargs):
        """
        Register a new face with embeddings from multiple angles.
        """
        # Validate input
        serializer = FaceImageSerializer(data=request.data)
        if not serializer.is_valid():
            print(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get validated data
            data = serializer.validated_data
            name = data.get('name')
            print(f"\n=== STARTING REGISTRATION FOR: {name} ===")

            # Initialize InsightFace
            app = get_face_app()
            print(f"InsightFace model loaded: {app}")

            # Dictionary to store embeddings by angle
            embeddings_by_angle = {}
            failed_angles = []

            # Process each angle
            for field_name, angle_name in FaceImageSerializer.ANGLE_MAPPING.items():
                print(f"\n--- Processing angle: {angle_name} (field: {field_name}) ---")
                
                # Get the base64 image data
                base64_data = data.get(field_name, '')
                
                if not base64_data:
                    print(f"Skipping {field_name}: No data provided")
                    continue

                try:
                    # Decode base64 to image
                    print(f"Decoding base64 image...")
                    image = FaceImageSerializer.decode_image(base64_data)
                    print(f"Image shape: {image.shape}")

                    # Detect faces
                    print(f"Detecting faces...")
                    faces = app.get(image)
                    print(f"Faces detected: {len(faces) if faces else 0}")
                    
                    if not faces or len(faces) == 0:
                        print(f"ERROR: No face detected in {angle_name}")
                        failed_angles.append(angle_name)
                        continue

                    # Get largest face
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    print(f"Selected face with bbox area: {(face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])}")

                    # Extract embedding
                    embedding = face.embedding.astype(np.float32)
                    print(f"Embedding shape: {embedding.shape}")
                    print(f"Embedding dtype: {embedding.dtype}")
                    print(f"Embedding sample (first 5): {embedding[:5]}")
                    
                    # Store as list
                    embeddings_by_angle[angle_name] = embedding.tolist()
                    print(f"✓ Successfully processed {angle_name}")

                except Exception as e:
                    print(f"ERROR processing {angle_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    failed_angles.append(f"{angle_name} ({str(e)})")
                    continue

            # Check if we have required angles
            required = {'frontal', 'top', 'bottom'}
            processed = set(embeddings_by_angle.keys())
            print(f"\n--- Validation ---")
            print(f"Required angles: {required}")
            print(f"Processed angles: {processed}")
            
            if not required.issubset(processed):
                missing = required - processed
                print(f"ERROR: Missing required angles: {missing}")
                return Response(
                    {
                        "error": f"Missing required angles: {missing}",
                        "failed_angles": failed_angles,
                        "processed_angles": list(processed)
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Calculate average embedding
            print(f"\n--- Averaging Embeddings ---")
            embeddings_list = [np.array(emb, dtype=np.float32) for emb in embeddings_by_angle.values()]
            print(f"Number of embeddings to average: {len(embeddings_list)}")
            print(f"First embedding shape: {embeddings_list[0].shape}")
            
            averaged_embedding = np.mean(embeddings_list, axis=0).astype(np.float32)
            print(f"Averaged embedding shape: {averaged_embedding.shape}")
            print(f"Averaged embedding sample (first 5): {averaged_embedding[:5]}")
            print(f"Contains NaN: {np.any(np.isnan(averaged_embedding))}")
            print(f"All zeros: {np.all(averaged_embedding == 0)}")

            # Convert to Python list
            print(f"\n--- Converting to JSON-safe format ---")
            array_as_list = [float(x) for x in averaged_embedding.tolist()]
            print(f"Final array length: {len(array_as_list)}")
            print(f"Final array sample (first 5): {array_as_list[:5]}")
            print(f"Array type: {type(array_as_list)}")
            print(f"Element type: {type(array_as_list[0])}")

            # Convert all angle embeddings
            angles_dict = {}
            for angle_name, embedding in embeddings_by_angle.items():
                embedding_list = [float(x) for x in np.array(embedding, dtype=np.float32).tolist()]
                angles_dict[angle_name] = embedding_list
                print(f"Stored {angle_name}: length={len(embedding_list)}")

            # Save to database
            print(f"\n--- Saving to Database ---")
            face_map_obj, created = FaceMap.objects.update_or_create(
                name=name,
                defaults={
                    'array': array_as_list,
                    'angles': angles_dict
                }
            )
            print(f"Saved to database: ID={face_map_obj.id}, created={created}")

            # Verify save
            print(f"Database verification:")
            print(f"  array type: {type(face_map_obj.array)}")
            print(f"  array length: {len(face_map_obj.array) if face_map_obj.array else 0}")
            print(f"  array sample: {face_map_obj.array[:5] if face_map_obj.array else 'EMPTY'}")
            print(f"  angles keys: {list(face_map_obj.angles.keys()) if face_map_obj.angles else 'EMPTY'}")

            # Return response
            return Response(
                {
                    "message": "Face registered successfully" if created else "Face updated successfully",
                    "id": face_map_obj.id,
                    "name": name,
                    "angles_processed": list(embeddings_by_angle.keys()),
                    "embedding_count": len(embeddings_by_angle),
                    "failed_angles": failed_angles if failed_angles else []
                },
                status=status.HTTP_201_CREATED if created else status.HTTP_200_OK
            )

        except Exception as e:
            print(f"\n!!! CRITICAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Registration failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class Recognize(viewsets.ViewSet):
    """
    ViewSet for face recognition.
    
    POST /recognize/ - Recognize a face from a single image
    Expected JSON:
    {
        "image": "base64_encoded_image",
        "threshold": 0.4 (optional, default 0.4)
    }
    """
    
    def create(self, request):
        """
        Recognize a face by comparing against stored embeddings.
        
        Returns:
        {
            "name": "Person Name" or "Unknown",
            "confidence": 0.95,
            "distance": 0.25,
            "match_found": true/false,
            "threshold_used": 0.4
        }
        """
        serializer = RecognizeSerializer(data=request.data)
        if not serializer.is_valid():
            print(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            print("\n=== STARTING FACE RECOGNITION ===")
            app = get_face_app()
            data = serializer.validated_data
            image_base64 = data['image']
            threshold = data.get('threshold', 0.4)

            print(f"Threshold: {threshold}")

            # Decode image
            print("Decoding base64 image...")
            image = RecognizeSerializer.decode_image(image_base64)
            print(f"Image shape: {image.shape}")

            # Detect faces and extract embedding
            print("Detecting faces...")
            faces = app.get(image)
            print(f"Faces detected: {len(faces) if faces else 0}")
            
            if not faces:
                print("No faces detected!")
                return Response(
                    {
                        "name": "Unknown",
                        "confidence": 0.0,
                        "distance": 999.0,
                        "match_found": False,
                        "error": "No face detected in the provided image.",
                        "threshold_used": threshold
                    },
                    status=status.HTTP_200_OK
                )

            # Use largest face by area
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = face.embedding.astype(np.float32)
            print(f"Embedding extracted: shape {embedding.shape}, dtype {embedding.dtype}")
            print(f"Embedding sample (first 5): {embedding[:5]}")

            # Get all stored faces and find best match
            print("\nComparing against stored faces...")
            stored_faces = FaceMap.objects.all()
            print(f"Total stored faces: {stored_faces.count()}")
            
            best_name = "Unknown"
            best_distance = 999.0  # Use large number instead of inf
            best_match_found = False

            for stored_face in stored_faces:
                print(f"\n  Comparing with: {stored_face.name}")
                
                # Skip empty arrays
                if not stored_face.array or len(stored_face.array) == 0:
                    print(f"    Skipping: array is empty")
                    continue
                
                print(f"    Array length: {len(stored_face.array)}")
                stored_embedding = np.array(stored_face.array, dtype=np.float32)
                distance = cosine(embedding, stored_embedding)
                print(f"    Cosine distance: {distance:.6f}")
                
                if distance < best_distance:
                    best_distance = distance
                    best_name = stored_face.name
                    best_match_found = distance < threshold
                    print(f"    ✓ New best match! Match found: {best_match_found}")

            # If no match within threshold, return Unknown
            if not best_match_found:
                best_name = "Unknown"
                print(f"\nBest distance {best_distance:.6f} exceeds threshold {threshold}")

            # Calculate confidence
            confidence = max(0.0, 1.0 - best_distance) if best_distance < 999.0 else 0.0
            
            print(f"\n=== RECOGNITION RESULT ===")
            print(f"Name: {best_name}")
            print(f"Distance: {best_distance:.6f}")
            print(f"Confidence: {confidence:.6f}")
            print(f"Match found: {best_match_found}")

            return Response(
                {
                    "name": best_name,
                    "confidence": float(confidence),
                    "distance": float(best_distance),
                    "match_found": best_match_found,
                    "threshold_used": threshold
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            import traceback
            print(f"\n!!! RECOGNITION ERROR !!!")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            return Response(
                {"error": f"Recognition failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class Stream(viewsets.ViewSet):
    """Stream live video with face-detection overlays over MJPEG."""

    def create(self, request):
        """
        POST /stream/

        Body: { "rtsp_url": "rtsp://..." }

        Returns a multipart/x-mixed-replace MJPEG stream.
        Green boxes = recognised faces, red boxes = unknown faces.
        The stream is cleanly shut down when the client disconnects.
        """
        rtsp_url = request.data.get('rtsp_url')
        if not rtsp_url:
            return Response(
                {"error": "Missing rtsp_url parameter"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            print(f"\n=== STARTING STREAM FROM {rtsp_url} ===")
            processor  = get_or_create_stream_processor(rtsp_url)
            stop_event = threading.Event()

            def on_close():
                """Called by Django when the client disconnects."""
                print(f"[Stream] Client disconnected — stopping generator for {rtsp_url}")
                stop_event.set()

            response = StreamingHttpResponse(
                processor.stream_generator(stop_event),
                content_type='multipart/x-mixed-replace; boundary=frame',
            )
            response.close = on_close  # hook into Django's response lifecycle
            return response

        except Exception as e:
            import traceback
            print(f"\n!!! STREAM ERROR: {e}")
            traceback.print_exc()
            return Response(
                {"error": f"Stream failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )