import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import os
import sys
import threading
from scipy.spatial.distance import cosine

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
RTSP_URL      = "rtsp://admin:trace321@192.168.100.64:554/Streaming/Channels/101"
DB_PATH       = "faces.npz"
MATCH_THRESH  = 0.4          # cosine distance threshold
RECOG_FPS     = 24         # max recognition passes per second
RECONNECT_SEC = 3            # seconds to wait before reconnecting

# ─────────────────────────────────────────────
# Face database helpers
# ─────────────────────────────────────────────

def load_db(path: str) -> dict:
    """Return {name: embedding_array} from .npz, or empty dict."""
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    return {str(k): data[k] for k in data.files}


def save_db(path: str, db: dict) -> None:
    np.savez(path, **db)


def match_face(embedding: np.ndarray, db: dict) -> tuple[str, float]:
    """Return (best_name, best_distance). Name is 'Unknown' if no match."""
    best_name = "Unknown"
    best_dist = float("inf")
    for name, stored_emb in db.items():
        dist = cosine(embedding, stored_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_dist >= MATCH_THRESH:
        best_name = "Unknown"
    return best_name, best_dist


# ─────────────────────────────────────────────
# Stream reader (threaded, with reconnection)
# ─────────────────────────────────────────────

class RTSPStream:
    def __init__(self, url: str):
        self.url   = url
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open_cap(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not self._stop:
            print(f"[Stream] Connecting to {self.url} …")
            cap = self._open_cap()
            if not cap.isOpened():
                print(f"[Stream] Failed to open. Retrying in {RECONNECT_SEC}s …")
                time.sleep(RECONNECT_SEC)
                continue
            print("[Stream] Connected.")
            consecutive_fails = 0
            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    consecutive_fails += 1
                    if consecutive_fails > 10:
                        print("[Stream] Stream lost. Reconnecting …")
                        break
                    time.sleep(0.05)
                    continue
                consecutive_fails = 0
                with self.lock:
                    self.frame = frame
            cap.release()
            if not self._stop:
                time.sleep(RECONNECT_SEC)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._stop = True


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────

def draw_face(frame, bbox, name: str, dist: float, matched: bool):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 220, 0) if matched else (0, 0, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{name}  d={dist:.2f}" if dist != float("inf") else name
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    txt = f"FPS: {fps:.1f}"
    cv2.putText(frame, txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Interactive terminal commands
# ─────────────────────────────────────────────

def cmd_delete(db: dict) -> dict:
    if not db:
        print("[Delete] Database is empty.")
        return db
    print("  Stored faces:", ", ".join(db.keys()))
    name = input("  Enter name to delete: ").strip()
    if name in db:
        del db[name]
        save_db(DB_PATH, db)
        print(f"[Delete] Removed '{name}'.")
    else:
        print(f"[Delete] '{name}' not found.")
    return db


def cmd_list(db: dict):
    if not db:
        print("[List] Database is empty.")
    else:
        print(f"[List] {len(db)} face(s) stored:", ", ".join(db.keys()))

def cmd_register(db: dict, app, frame) -> dict:
    """Register a face from the current frame by printing its embedding."""
    faces = app.get(frame)
    if not faces:
        print("[Register] No face detected in current frame.")
        return db
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = face.embedding
    print("[Register] Current face embedding (512-d list, copy this):")
    print(list(emb))
    print("[Register] Copy this list and use 'M' to manually register with a name, or hardcode it in the script.\n")
    return db


def cmd_manual_register(db: dict) -> dict:
    """Manually register a face by pasting an embedding array."""
    name = input("  Enter name for the face: ").strip()
    if not name:
        print("[Manual Register] Name cannot be empty.")
        return db
    if name in db:
        overwrite = input(f"  '{name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("[Manual Register] Cancelled.")
            return db
    print("  Paste the 512-d embedding array (e.g., np.array([...])):")
    emb_input = input("  ").strip()
    try:
        # Evaluate the input as a numpy array
        emb = eval(emb_input, {"np": np, "array": np.array})
        if not isinstance(emb, np.ndarray) or emb.shape != (512,):
            raise ValueError("Invalid shape or type.")
        db[name] = emb.astype(np.float32)
        save_db(DB_PATH, db)
        print(f"[Manual Register] Registered '{name}'.")
    except Exception as e:
        print(f"[Manual Register] Error parsing embedding: {e}")
    return db

# ─────────────────────────────────────────────
# Embedding capture helpers
# ─────────────────────────────────────────────

class EmbeddingCapture:
    """Capture and average embeddings from multiple head angles."""
    def __init__(self):
        self.embeddings = {}
        self.angles = ['frontal', 'left', 'right', 'up', 'down']
    
    def capture(self, app, frame, angle: str) -> bool:
        """Capture embedding for a given angle. Return True if successful."""
        faces = app.get(frame)
        if not faces:
            print(f"[Capture] No face detected for {angle}.")
            return False
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        self.embeddings[angle] = face.embedding
        print(f"[Capture] Captured {angle}: shape={face.embedding.shape}")
        return True
    
    def average(self) -> tuple[np.ndarray, dict]:
        """Return averaged embedding and capture summary."""
        if not self.embeddings:
            return None, {}
        
        embeddings_list = list(self.embeddings.values())
        avg_embedding = np.mean(embeddings_list, axis=0).astype(np.float32)
        return avg_embedding, self.embeddings
    
    def get_status(self) -> str:
        """Return capture status."""
        captured = ", ".join(self.embeddings.keys())
        remaining = ", ".join(a for a in self.angles if a not in self.embeddings)
        return f"Captured: {captured or 'none'} | Remaining: {remaining or 'complete'}"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("[Init] Loading InsightFace buffalo_l model …")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[Init] Model ready.")

    db = load_db(DB_PATH)
    
    # Hardcoded Muby embedding
    muby_embedding = np.array([-0.5651025, 0.36092085, 0.10796185, -0.06164473, -1.4997828, -0.55010915,
  0.15529194, 2.1174984, -0.06378295, 1.4612863, -1.0071673, -0.9407381,
 -0.04653333, -0.52508414, -0.9408227, 0.74579716, -0.113657, -0.57020056,
 -0.28095907, -0.7708045, 1.0091213, -0.18131503, 0.6650626, -0.10880103,
 -0.12772052, 0.37112966, -0.48924404, 0.06994317, 0.03148569, -1.9474618,
  0.7802154, -0.6968235, 0.11434969, -0.05710487, 0.22514, -0.35713792,
 -1.1560234, 0.7134832, -0.22210567, 1.3683926, -0.23812814, -0.82320577,
  0.28497666, 0.23395813, 0.97858256, 0.41640073, 1.8870385, 1.1506987,
  0.29804522, 0.8570217, 0.38614392, 0.73914737, -0.133558, -0.40469798,
  0.18062074, -0.7619716, -0.14587939, 0.66997266, 0.8292376, -1.1758684,
 -0.770816, -1.2197021, -0.4356423, 0.92748773, 0.0224176, -1.0331488,
  0.13243362, -0.26657358, 1.3823261, -0.13285545, -0.0304011, -1.323858,
  0.24381916, -0.8146367, -0.62767416, 0.13646482, 2.4620197, 0.4903946,
  0.04637239, -1.2521746, -1.6519953, -0.41402927, 0.5508019, -0.24057174,
  0.4449502, -0.25163987, 1.0255483, 0.6286065, -0.84047335, 0.11035623,
  0.5383858, -0.45827612, -0.71121055, -0.8740964, 0.54191756, 0.48083347,
 -0.62656635, 0.39204058, 0.02206349, 0.5557435, -0.9416669, 0.05954264,
 -0.49418098, 0.6746939, 0.49734616, 1.4616424, 1.3508303, -1.4335779,
 -0.8317607, 0.19024174, -0.04110942, 0.17032373, -1.6273695, 0.6730638,
  1.6640829, 1.7333758, 0.13702527, -1.1527154, -0.11045561, -0.4266416,
  0.11096253, -0.0113137, -1.7364428, -1.4955595, 0.7512547, -0.94082963,
  0.7846792, 0.85338056, -0.4354174, 0.22924562, -0.24128711, -1.0842167,
 -0.06616246, -0.3056422, 0.3357582, 0.76662034, -0.7695598, -0.5142984,
 -1.0516351, -1.2718138, -0.9226204, -0.7683666, 0.69903135, -1.5394624,
  1.7293361, -1.8131607, 0.5373186, 1.1439284, -1.5041778, 0.07588486,
 -0.27863607, 0.59054476, -0.88567483, -1.4441897, -1.6632382, -0.9938471,
 -0.861361, -0.94425106, -0.4034558, 0.26788765, 0.37393647, -1.7110243,
 -0.70050436, -0.07115866, 0.7949394, 0.28348127, 1.3356272, 0.55952764,
  0.60519683, -0.19844487, 0.07626238, 0.5378781, -0.650489, -1.24003,
  0.11666004, 1.1946831, 0.5141853, -0.6898189, -0.08507065, 1.0765632,
 -0.4399251, -0.32208323, 0.7087034, -0.11563251, 0.09449402, -0.75145394,
  0.6825392, -1.2446026, -0.58340484, -0.40099207, 0.34004587, 2.0014658,
 -0.50509393, 0.08800532, 0.888471, -0.16697355, 0.34064436, 0.05590091,
  0.77925384, -1.4232695, 0.13401732, 1.1594031, -0.15800193, -1.1126629,
  1.2830932, -0.10167952, -0.4058655, -0.84491587, 0.19379607, 0.0116095,
  0.41356468, -0.8088134, 0.9710514, -0.2924816, -1.0320966, -0.05858835,
  1.2071472, -0.22939841, -0.04800241, -0.82389945, 1.4913286, 0.96260643,
  0.3347912, 0.8590956, 0.40722045, 0.16637097, 1.1761614, 0.78583777,
 -0.52535325, 0.31650788, 0.44410935, 0.03669427, 1.1270523, -0.06767897,
  1.1323918, 1.173119, -1.2549784, 0.531601, 0.41730076, 0.50198793,
  0.5241337, -0.42508584, -1.9108483, 0.13192889, -0.6388851, -0.16882403,
 -0.34656897, 0.32505357, 1.5753973, -0.9683129, 1.1533787, 0.04779353,
  0.49561602, 0.05896559, 0.54132044, -1.3093517, -0.35054055, -0.20513196,
 -0.41792387, 1.4616015, -0.28114337, -0.3563811, 1.4370453, -1.0239747,
 -0.48158512, 0.800068, 0.13053687, -0.01643685, 1.2414231, 0.2913196,
  2.011576, -0.37363288, 1.178744, 0.24859318, -0.18868458, -0.2881938,
 -0.35327443, -0.05250089, -0.7162017, -0.22295003, -0.19485924, 0.4336872,
 -0.00957382, -0.26217538, 0.95724964, 1.3064371, -0.9265603, -0.5689089,
  0.11761723, -0.51893747, -1.7688551, -0.25846958, 1.1146233, -0.15620926,
  0.6978321, -0.2524715, 0.9494643, -0.42180568, 0.5293051, 0.99897623,
  2.3481636, 0.6785758, -0.35397536, 0.6593937, -0.6223229, -0.9412602,
 -0.20755942, -0.35152045, 0.16899085, -0.21646306, -0.0149526, -0.97632253,
  1.9353737, 1.6025598, -1.0331208, 0.37302, -0.6626264, -1.0980818,
  0.8136306, -0.09717879, -0.04233104, -1.701181, -1.7304819, 0.21472816,
 -0.53121245, -0.04886333, 0.6879419, -1.1445115, 1.0629619, 0.25730044,
 -0.6563524, 0.20845059, 0.5846573, 0.04152388, -0.27770537, -0.84389174,
  0.05327928, 0.7627536, 0.7159162, -0.55999327, -0.3544931, 0.30740345,
  1.1217645, -0.11984082, -0.5603305, -0.20507959, -1.4742801, -0.6184487,
  1.0160139, 1.0735247, -0.12171638, -0.31314978, -1.0297703, -0.09820386,
 -0.60516506, -0.9280955, -0.69093525, 0.08170122, 0.28722423, 0.6961377,
  0.7371246, 1.128882, -0.0043036, 1.0716377, -0.6184114, 0.12642841,
  1.8216722, -0.45372352, -0.23228428, -0.16116838, 0.20091519, -0.47028312,
 -0.70032233, 0.8572974, 0.38792497, -0.1281256, 0.59866345, 0.09154109,
  0.6614696, 0.88685817, -1.4034114, -1.25982, 0.1482789, -0.9078108,
 -0.38655892, 0.21337657, 0.45876646, -0.5555302, -1.0966394, -0.53285533,
  0.1500837, -0.57881254, -0.20469324, -0.68851227, -0.998106, -0.47319755,
 -1.0087503, 1.4626383, -0.5420911, 1.5860984, 1.8536412, 0.6153568,
  0.68489444, -1.0211279, 0.21339159, 0.39514494, 0.04621474, 0.39572853,
  1.4001637, 0.8600007, -0.43889338, 1.3492372, 0.757476, -0.7986733,
 -0.3445549, 0.5961092, -0.3581628, 0.38315308, 0.74527293, 0.43083492,
  1.2055966, 0.33232817, -0.17874241, 0.679022, 0.74212945, 1.3547373,
 -0.13746533, 0.1359572, 1.2198007, 0.70289886, -0.56725144, -0.18642512,
  1.0802562, -2.1223, -1.4375868, -0.60432345, 0.4184424, -0.16752212,
  1.8179939, 0.3147524, 0.59864527, -0.34115478, 0.07793976, -0.7874758,
 -0.6560129, -0.35510916, -0.41108784, -1.2847592, 0.2699319, -0.524969,
 -0.18691537, 0.10542073, 0.30447215, 0.37790626, -0.5037708, 0.14771374,
  0.12234779, 0.12535797, 0.7504895, -1.2106708, 0.67956054, 0.8279356,
  0.2549841, -0.7141105, -0.02873972, -1.1579497, -0.12137499, -0.35903096,
 -0.20954423, 1.1655915, -0.18962777, 1.6496083, 0.42542267, 0.23289756,
 -0.11923528, 0.27939114, -0.4244341, 0.11398909, -0.4884057, -0.4067501,
  0.0686473, -0.43922204, -0.3479205, -0.3987022, -1.0538566, -1.6440563,
  0.78963053, -0.10892687, -0.20088784, 0.18873186, 0.44513327, -0.79498404,
  0.5945609, -1.2455778, -0.04220954, -0.8103961, -0.5716684, 1.1148434,
 -1.4639256, -1.6286503, 0.26180533, 0.7274621, 0.13703716, 1.1775283,
 -0.4172755, -0.54373205, -0.11815486, -1.4519377, 0.85893553, -0.11299171,
  0.3285988, 0.5394645], dtype=np.float32)
    
    db["Muby"] = muby_embedding
    print(f"[DB] Loaded {len(db)} face(s) from {DB_PATH} + Muby and Isaac embeddings.")

    # Hardcode additional embeddings here, e.g.:
    isaac_embedding = np.array([np.float32(-0.7299673), np.float32(1.3744767), np.float32(0.89397395), np.float32(1.2039645), np.float32(-0.04794973), np.float32(-0.54136026), np.float32(1.1325452), np.float32(-0.7357819), np.float32(0.8671769), np.float32(0.53870475), np.float32(-0.42634487), np.float32(-0.46847945), np.float32(0.42543054), np.float32(1.8843338), np.float32(0.5947447), np.float32(-0.0006214857), np.float32(0.054740477), np.float32(-0.4509483), np.float32(-0.4989868), np.float32(0.839907), np.float32(-0.899103), np.float32(1.0473986), np.float32(0.4518959), np.float32(0.40156946), np.float32(-0.7090035), np.float32(-0.5242143), np.float32(0.2616306), np.float32(0.02767188), np.float32(0.93066347), np.float32(0.28039116), np.float32(-1.0544736), np.float32(-1.1390792), np.float32(0.2678494), np.float32(-0.85513383), np.float32(-0.6182897), np.float32(-0.54464257), np.float32(-1.8791971), np.float32(0.31875977), np.float32(0.4210177), np.float32(0.8473266), np.float32(-1.4198472), np.float32(0.11074734), np.float32(0.30598313), np.float32(0.011989169), np.float32(1.2617568), np.float32(0.51135784), np.float32(-1.5035906), np.float32(-0.17448762), np.float32(-1.0266101), np.float32(0.2337871), np.float32(1.0420554), np.float32(-1.8295186), np.float32(-0.93293494), np.float32(0.64940685), np.float32(-0.59371316), np.float32(-0.34253088), np.float32(-1.0465713), np.float32(-0.4081471), np.float32(1.9384239), np.float32(1.1373419), np.float32(0.29850107), np.float32(0.5530921), np.float32(0.86475337), np.float32(0.08779226), np.float32(-2.0075078), np.float32(0.5681367), np.float32(-0.3103822), np.float32(-0.30553454), np.float32(0.9146741), np.float32(0.028047955), np.float32(1.0912244), np.float32(0.95245665), np.float32(-1.5241592), np.float32(-0.45530653), np.float32(0.41733783), np.float32(1.6351118), np.float32(-0.050832093), np.float32(-0.62333935), np.float32(-1.7058773), np.float32(0.013017851), np.float32(1.4445086), np.float32(-0.5323659), np.float32(0.73224145), np.float32(-0.18327188), np.float32(-0.17263749), np.float32(0.9723814), np.float32(0.5076125), np.float32(0.2694494), np.float32(-0.5733402), np.float32(0.41015887), np.float32(-0.6323583), np.float32(2.1820664), np.float32(0.8725996), np.float32(-0.07723093), np.float32(1.1319582), np.float32(1.5747563), np.float32(-0.98743117), np.float32(-0.45567614), np.float32(0.446379), np.float32(1.6241783), np.float32(-0.2967136), np.float32(0.2424305), np.float32(0.012042021), np.float32(0.41397724), np.float32(-0.6047081), np.float32(-0.4114863), np.float32(0.41443068), np.float32(0.109628834), np.float32(0.23906545), np.float32(0.5498405), np.float32(0.5702013), np.float32(-1.0393648), np.float32(1.6611353), np.float32(-0.53348464), np.float32(-1.4283696), np.float32(0.17128591), np.float32(-0.48055547), np.float32(2.0514627), np.float32(-1.6147296), np.float32(1.1270936), np.float32(0.27543932), np.float32(0.5852153), np.float32(0.0692349), np.float32(-0.5905587), np.float32(-0.2509664), np.float32(-0.6637613), np.float32(-0.11839595), np.float32(-0.18941693), np.float32(-1.827383), np.float32(0.10034766), np.float32(-1.0613015), np.float32(-0.8440153), np.float32(-1.683486), np.float32(-0.71131223), np.float32(-0.28184366), np.float32(0.26609185), np.float32(-0.72879297), np.float32(-0.5926857), np.float32(0.33833346), np.float32(0.53633356), np.float32(0.20979209), np.float32(0.5556961), np.float32(0.24131355), np.float32(-0.27456757), np.float32(0.15190697), np.float32(-0.43618226), np.float32(-0.67421454), np.float32(-0.27350467), np.float32(-0.72563875), np.float32(1.6408501), np.float32(0.013439322), np.float32(-1.4224415), np.float32(-0.03176987), np.float32(1.157006), np.float32(-1.1980513), np.float32(-0.5339816), np.float32(1.6015205), np.float32(0.96027696), np.float32(-1.2312524), np.float32(0.56586015), np.float32(-0.29319802), np.float32(-1.0965135), np.float32(0.36997092), np.float32(0.70206654), np.float32(-0.15201044), np.float32(1.0266705), np.float32(-0.38872308), np.float32(1.013065), np.float32(0.20251563), np.float32(-0.94198114), np.float32(-0.19262724), np.float32(0.5830573), np.float32(0.32727012), np.float32(1.2122322), np.float32(0.0052704336), np.float32(-0.1445488), np.float32(1.3855808), np.float32(-0.5056658), np.float32(0.29620558), np.float32(1.8828586), np.float32(-0.01207563), np.float32(1.0856848), np.float32(-0.2594211), np.float32(-1.0590026), np.float32(-0.88185215), np.float32(-0.79253906), np.float32(0.19314402), np.float32(1.1624042), np.float32(1.0871891), np.float32(-0.2115601), np.float32(-0.5893862), np.float32(1.1272122), np.float32(-0.85555106), np.float32(-0.18234114), np.float32(0.9728346), np.float32(0.5807112), np.float32(-0.4537236), np.float32(0.6290356), np.float32(-0.119515635), np.float32(-1.6956728), np.float32(0.84537256), np.float32(-0.067418136), np.float32(0.29135078), np.float32(-1.2718054), np.float32(0.12839146), np.float32(0.40562695), np.float32(0.9359978), np.float32(-1.0100852), np.float32(-0.24626522), np.float32(-1.0899156), np.float32(-0.49332613), np.float32(1.2921724), np.float32(0.3594727), np.float32(0.024304008), np.float32(0.78353816), np.float32(1.6989787), np.float32(0.9262535), np.float32(-0.22276005), np.float32(-0.20923047), np.float32(0.008277726), np.float32(0.93865645), np.float32(0.3470284), np.float32(-0.3518509), np.float32(0.22394828), np.float32(-1.1019002), np.float32(0.23176876), np.float32(-0.8298917), np.float32(0.89309824), np.float32(0.14229165), np.float32(1.3075055), np.float32(0.42315692), np.float32(1.6454275), np.float32(-0.33314523), np.float32(0.23942132), np.float32(0.29978505), np.float32(-0.99095166), np.float32(0.46627888), np.float32(0.67121726), np.float32(-1.8698019), np.float32(0.0010028065), np.float32(-1.4185479), np.float32(-0.93570757), np.float32(-1.4106398), np.float32(0.077358946), np.float32(-0.8881046), np.float32(-0.7630661), np.float32(0.048732124), np.float32(1.7650547), np.float32(0.07455591), np.float32(1.1831524), np.float32(-0.10339481), np.float32(1.2041304), np.float32(-0.3642053), np.float32(-1.8656833), np.float32(-0.060265172), np.float32(0.7870087), np.float32(1.675915), np.float32(-0.76476467), np.float32(1.2936969), np.float32(0.6648595), np.float32(1.0191371), np.float32(-1.2101353), np.float32(0.4784926), np.float32(0.09834568), np.float32(0.17458618), np.float32(0.92426574), np.float32(-0.6371615), np.float32(0.05223316), np.float32(-1.0499281), np.float32(0.8405083), np.float32(0.22554211), np.float32(-0.62958), np.float32(0.04097767), np.float32(-0.23891301), np.float32(0.39288607), np.float32(0.74157006), np.float32(-1.6315367), np.float32(0.56248385), np.float32(1.3510822), np.float32(-0.59606034), np.float32(0.8970667), np.float32(0.2588188), np.float32(0.42097324), np.float32(0.070645355), np.float32(-0.2021285), np.float32(0.7513037), np.float32(-2.3820074), np.float32(0.019406458), np.float32(1.1652057), np.float32(-1.497868), np.float32(-0.8962386), np.float32(-0.43671346), np.float32(0.63293433), np.float32(-0.60700524), np.float32(0.625432), np.float32(0.30765408), np.float32(-0.37738112), np.float32(0.3877645), np.float32(0.31743896), np.float32(-1.3934419), np.float32(0.26611358), np.float32(-0.32359892), np.float32(-0.726038), np.float32(0.88276136), np.float32(0.82999974), np.float32(0.624004), np.float32(0.2763893), np.float32(0.1961514), np.float32(-1.6985781), np.float32(0.8635298), np.float32(-0.6172239), np.float32(-0.42971689), np.float32(0.6856047), np.float32(-1.1961505), np.float32(0.5792925), np.float32(0.3561565), np.float32(-0.16873185), np.float32(0.87779856), np.float32(-1.1942215), np.float32(1.3754747), np.float32(1.3455656), np.float32(-1.2258584), np.float32(-0.59035206), np.float32(0.5390145), np.float32(-1.0776557), np.float32(0.93337345), np.float32(0.45053124), np.float32(-0.79996264), np.float32(0.47722712), np.float32(0.77752125), np.float32(-0.033622794), np.float32(0.6499115), np.float32(0.19853431), np.float32(0.937113), np.float32(-0.3514455), np.float32(-0.37932795), np.float32(-0.34335408), np.float32(-0.40019256), np.float32(1.3657745), np.float32(0.2111874), np.float32(1.1660467), np.float32(-1.2690182), np.float32(-0.895978), np.float32(-0.6312148), np.float32(1.2181664), np.float32(0.1001796), np.float32(0.37190598), np.float32(-0.25893536), np.float32(-0.05998396), np.float32(-0.7510558), np.float32(-0.71379316), np.float32(0.29440743), np.float32(0.99456203), np.float32(0.92450917), np.float32(-1.8343542), np.float32(-1.4040921), np.float32(-1.4742725), np.float32(-0.059114862), np.float32(0.2004786), np.float32(1.2335677), np.float32(0.113835335), np.float32(-0.8887779), np.float32(-0.62342227), np.float32(-0.06330595), np.float32(0.59532696), np.float32(0.5809982), np.float32(0.06782712), np.float32(0.0016085387), np.float32(0.21905608), np.float32(0.94594896), np.float32(0.3281988), np.float32(0.922798), np.float32(-1.2834378), np.float32(1.3031574), np.float32(-0.5483478), np.float32(-0.48818558), np.float32(-0.50963837), np.float32(0.34712344), np.float32(0.52504903), np.float32(-1.4385006), np.float32(0.2665708), np.float32(-0.4760531), np.float32(-0.32982194), np.float32(-0.36867902), np.float32(-0.047417045), np.float32(-0.54096), np.float32(-0.018467773), np.float32(0.22084692), np.float32(1.2697133), np.float32(0.8805868), np.float32(-0.17550567), np.float32(-1.3302133), np.float32(0.44010314), np.float32(1.0021069), np.float32(0.023496974), np.float32(-0.89719784), np.float32(0.57607865), np.float32(0.4861628), np.float32(-0.29897767), np.float32(0.28843665), np.float32(-0.54938424), np.float32(-0.47979698), np.float32(-0.88985765), np.float32(-1.0339018), np.float32(-0.8965128), np.float32(-0.7752626), np.float32(-1.2378652), np.float32(-0.5573969), np.float32(-1.9578604), np.float32(-0.1052532), np.float32(-0.42146474), np.float32(-0.084273785), np.float32(0.68841416), np.float32(-1.3217648), np.float32(-0.12358679), np.float32(-0.6700047), np.float32(0.011198103), np.float32(-1.2856585), np.float32(-1.0120411), np.float32(-0.69805557), np.float32(-0.061312072), np.float32(-0.04783987), np.float32(0.7221104), np.float32(0.030303013), np.float32(-0.31608778), np.float32(0.2021294), np.float32(0.2660548), np.float32(0.65451974), np.float32(-0.05256917), np.float32(0.93011475), np.float32(1.6295414), np.float32(0.03364138), np.float32(0.14642802), np.float32(-0.364502), np.float32(0.24132867), np.float32(1.5323253), np.float32(-1.3393071), np.float32(0.2985892), np.float32(-1.0815709), np.float32(0.6369836), np.float32(1.5689204), np.float32(0.68320584), np.float32(-1.4990782), np.float32(-0.30440083), np.float32(-0.8881651), np.float32(1.040836), np.float32(-0.45304027), np.float32(1.1197523), np.float32(0.6232674), np.float32(-0.25059426), np.float32(-0.25685775), np.float32(0.41612086), np.float32(-0.58131826), np.float32(-1.1623657), np.float32(-0.5134076), np.float32(-1.7159526), np.float32(0.2963388), np.float32(-0.16455807), np.float32(-0.03403324), np.float32(1.0534902), np.float32(0.6431023), np.float32(0.25739312), np.float32(0.82838696), np.float32(1.0106003), np.float32(-0.16295925), np.float32(0.15319788), np.float32(-0.2788337), np.float32(1.092988), np.float32(-0.7928301), np.float32(0.13526866), np.float32(-2.0757315), np.float32(0.45214564), np.float32(0.34046906), np.float32(-0.14659378), np.float32(1.2913922), np.float32(-1.1435697), np.float32(-0.3419072), np.float32(-0.41009617), np.float32(2.290283), np.float32(0.06335044), np.float32(0.22719376), np.float32(-0.020535242), np.float32(-1.7655346), np.float32(1.1353155), np.float32(-0.87561214), np.float32(-0.15055034), np.float32(0.16962853), np.float32(0.41816193), np.float32(-0.37135753), np.float32(0.92375517), np.float32(-0.84208024), np.float32(-0.93866825), np.float32(-0.237782), np.float32(0.968964), np.float32(-0.5246369), np.float32(-1.1140207), np.float32(1.5517229), np.float32(0.13129523), np.float32(-0.11262671), np.float32(-0.21630426), np.float32(-1.1356709), np.float32(-0.27582136), np.float32(0.48206544), np.float32(-0.9923134), np.float32(-0.51990134), np.float32(0.33531255), np.float32(-0.6030234), np.float32(-0.006880927), np.float32(2.2179854), np.float32(1.6921451), np.float32(0.09902611), np.float32(0.5496725)])
    db["Isaac"] = isaac_embedding
    print(f"[DB] Loaded {len(db)} face(s) from {DB_PATH} + Muby and Isaac embeddings.")

    stream = RTSPStream(RTSP_URL)

    # Recognition state
    last_recog_time  = 0.0
    recog_interval   = 1.0 / RECOG_FPS
    cached_results   = []          # list of (bbox, name, dist, matched)

    # FPS tracking
    fps_counter = 0
    fps_display = 0.0
    fps_timer   = time.time()

    print("\nControls:  C = capture multi-angle embedding  |  R = register current face (print embedding)  |  M = manual register (paste embedding)  |  D = delete face  |  L = list faces  |  Q = quit\n")

    cv2.namedWindow("RTSP Face Recognition", cv2.WINDOW_NORMAL)
    
    capture_mode = False
    embedding_capture = None

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.05)
            continue

        now = time.time()

        # ── Recognition pass (throttled to RECOG_FPS) ──────────────────
        if now - last_recog_time >= recog_interval:
            last_recog_time = now
            faces = app.get(frame)
            cached_results = []
            for face in faces:
                emb  = face.embedding
                name, dist = match_face(emb, db)
                matched = name != "Unknown"
                cached_results.append((face.bbox, name, dist, matched))

        # ── Draw cached detections on current frame ──────────────────────
        display = frame.copy()
        for bbox, name, dist, matched in cached_results:
            draw_face(display, bbox, name, dist, matched)

        # ── FPS ──────────────────────────────────────────────────────────
        fps_counter += 1
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = now
        draw_fps(display, fps_display)

        cv2.imshow("RTSP Face Recognition", display)

        # ── Key handling ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        
        if capture_mode:
            # In capture mode
            if key == ord('f') or key == ord('F'):
                if embedding_capture.capture(app, frame, 'frontal'):
                    print(f"  {embedding_capture.get_status()}")
            elif key == ord('l') or key == ord('L'):
                if embedding_capture.capture(app, frame, 'left'):
                    print(f"  {embedding_capture.get_status()}")
            elif key == ord('r') or key == ord('R'):
                if embedding_capture.capture(app, frame, 'right'):
                    print(f"  {embedding_capture.get_status()}")
            elif key == ord('u') or key == ord('U'):
                if embedding_capture.capture(app, frame, 'up'):
                    print(f"  {embedding_capture.get_status()}")
            elif key == ord('d') or key == ord('D'):
                if embedding_capture.capture(app, frame, 'down'):
                    print(f"  {embedding_capture.get_status()}")
            elif key == ord('a') or key == ord('A'):
                avg_emb, caps = embedding_capture.average()
                if avg_emb is not None:
                    print("\n[Average Embedding] 512-d vector (copy this list):")
                    print(list(avg_emb))
                    print(f"\n[Info] Captured {len(caps)} angles: {list(caps.keys())}\n")
                    print("[Instructions] Copy the list above and hardcode it in the script like: isaac_embedding = np.array([...])\n")
                    capture_mode = False
                    embedding_capture = None
                    print("[Capture] Exited capture mode.\n")
            elif key == 27:  # ESC
                print("[Capture] Cancelled.\n")
                capture_mode = False
                embedding_capture = None

        else:
            # Normal mode
            if key == ord('q') or key == ord('Q'):
                print("[Quit] Exiting …")
                break

            elif key == ord('c') or key == ord('C'):
                print("[Capture] Entering multi-angle embedding capture mode.")
                print("[Capture] Instructions:")
                print("  F = capture frontal")
                print("  L = capture left (30°)")
                print("  R = capture right (30°)")
                print("  U = capture up")
                print("  D = capture down")
                print("  A = average and print result")
                print("  ESC = cancel\n")
                capture_mode = True
                embedding_capture = EmbeddingCapture()

            elif key == ord('r') or key == ord('R'):
                db = cmd_register(db, app, frame)

            elif key == ord('m') or key == ord('M'):
                db = cmd_manual_register(db)

            elif key == ord('d') or key == ord('D'):
                db = cmd_delete(db)

            elif key == ord('l') or key == ord('L'):
                cmd_list(db)

    stream.stop()
    cv2.destroyAllWindows()
    print("[Done]")


if __name__ == "__main__":
    main()