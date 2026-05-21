import os
import json
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from dotenv import load_dotenv

# ─── Load .env before anything else ──────────────────────────────────────────
load_dotenv()

# ─── Password hashing ─────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ─── Config ───────────────────────────────────────────────────────────────────
# FIX [CRITICAL]: Credentials MUST come from .env — no hardcoded fallback.
# Create a .env file (see .env.example) before running.
MONGODB_URI   = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "wellness_analytics")

# FIX [WARN]: CORS origins from env, comma-separated
_raw_origins  = os.getenv("CORS_ORIGINS", "http://localhost:3000")
CORS_ORIGINS  = [o.strip() for o in _raw_origins.split(",")]

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
STORAGE_DIR    = os.getenv("STORAGE_DIR",    "storage_vault")

os.makedirs(STORAGE_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─── Runtime state ────────────────────────────────────────────────────────────
BODY_CLASSES    = ["Face", "Skin Hand", "Eye", "Forehead", "Cheek",
                   "Neck", "Arm", "Leg"]
DISEASE_CLASSES = ["Healthy Base", "Surface Fatigue",
                   "Dehydration Zone", "Vascular Flush"]
body_model    = None
disease_model = None
ae_model      = None
db            = None
users_col     = None
scans_col     = None
mongo_client  = None


# ─── Neural Network Architectures ─────────────────────────────────────────────
class BodyPartClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input: (B, 3, 224, 224)
        # After two MaxPool2d(2,2): → 56×56
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 56
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).view(x.size(0), -1))


class DiseaseClassifierWithGradCAM(nn.Module):
    """
    Input 224×224 spatial flow:
      conv1 + pool → 112   conv2 + pool → 56
      conv3 (no pool) → 56   [activations hook here]
      pool(conv3_out) → 28   fc: 64 * 28 * 28 = 50 176
    """
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        # FIX [BUG]: correct spatial dim is 28×28 (one pool after conv3)
        self.fc    = nn.Linear(64 * 28 * 28, num_classes)
        self.gradients  : Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def _activations_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 112
        x = self.pool(F.relu(self.conv2(x)))   # 56
        x = F.relu(self.conv3(x))              # 56 — capture here
        self.activations = x
        # FIX [BUG]: guard — only register hook when graph is being built
        if x.requires_grad:
            x.register_hook(self._activations_hook)
        x = self.pool(x)                        # 28
        return self.fc(x.view(x.size(0), -1))


class NormalAutoencoder(nn.Module):
    """
    Input: (B, 3, 56, 56)
      → stride-2 conv ×2 → 14×14  → flatten → latent(8)
      → decode back to 56×56
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=2, padding=1), nn.ReLU(),   # 28
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 14
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32 * 14 * 14), nn.ReLU(),
            nn.Unflatten(1, (32, 14, 14)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,  3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        latent = F.normalize(self.encoder(x), p=2, dim=1)
        return latent, self.decoder(latent)


# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────
# FIX [BUG]: on_event is deprecated since FastAPI 0.93; use lifespan instead.
@asynccontextmanager
async def lifespan(application: "FastAPI"):
    # ── STARTUP ──────────────────────────────────────────────────────────────
    global body_model, disease_model, ae_model
    global db, users_col, scans_col, mongo_client, BODY_CLASSES, DISEASE_CLASSES

    if not MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI is not set. Create a .env file — see .env.example."
        )

    print("🔗 Connecting to MongoDB …")
    mongo_client = AsyncIOMotorClient(
        MONGODB_URI,
        maxPoolSize=50,
        minPoolSize=10,
        maxIdleTimeMS=60_000,
        connectTimeoutMS=30_000,
        serverSelectionTimeoutMS=30_000,
    )
    try:
        await mongo_client.admin.command("ping")
        print("✅ MongoDB connected successfully.")
    except Exception as exc:
        raise RuntimeError(f"MongoDB connection failed: {exc}") from exc

    db        = mongo_client[DATABASE_NAME]
    users_col = db["users"]
    scans_col = db["scans"]

    await users_col.create_index("email",   unique=True, background=True)
    await users_col.create_index("user_id", unique=True, background=True)
    await scans_col.create_index("user_id",             background=True)
    await scans_col.create_index("scan_id", unique=True, background=True)
    await scans_col.create_index("timestamp",           background=True)

    # Instantiate models
    body_model    = BodyPartClassifier(num_classes=len(BODY_CLASSES)).to(DEVICE)
    disease_model = DiseaseClassifierWithGradCAM(num_classes=len(DISEASE_CLASSES)).to(DEVICE)
    ae_model      = NormalAutoencoder().to(DEVICE)

    # Load checkpoints if available
    try:
        mapping_path = os.path.join(CHECKPOINT_DIR, "class_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                mapping = json.load(f)
            BODY_CLASSES    = mapping["body_classes"]
            DISEASE_CLASSES = mapping["disease_classes"]

        for fname, model in [
            ("body_classifier.pth",    body_model),
            ("disease_classifier.pth", disease_model),
            ("patch_autoencoder.pth",  ae_model),
        ]:
            pth = os.path.join(CHECKPOINT_DIR, fname)
            if os.path.exists(pth):
                model.load_state_dict(torch.load(pth, map_location=DEVICE))
                print(f"  ✅ Loaded {fname}")
            else:
                print(f"  ⚠️  {fname} not found — using random weights (demo mode).")

    except Exception as exc:
        print(f"⚠️  Checkpoint load warning: {exc}  (running with random weights)")

    body_model.eval()
    disease_model.eval()
    ae_model.eval()
    print(f"🚀 App ready  |  device={DEVICE}")

    yield   # ← application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    if mongo_client:
        mongo_client.close()
        print("🔌 MongoDB connection closed.")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "BioCV Enterprise Wellness Analytics Core",
    version     = "2.1.0",
    description = "Backend pipeline with computer vision, MongoDB tracking, and Admin panels.",
    lifespan    = lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,  # FIX [WARN]: from env
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    full_name: str
    email    : str
    password : Optional[str] = None

class UserLogin(BaseModel):
    email   : str
    password: Optional[str] = None


# ─── CV helper functions ──────────────────────────────────────────────────────
def run_quality_checks(img: np.ndarray):
    gray             = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score       = int(cv2.Laplacian(gray, cv2.CV_64F).var())
    lab              = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    brightness_score = int(np.mean(cv2.split(lab)[0]))
    return blur_score, brightness_score


def compute_gradcam_patches(
    image_tensor: torch.Tensor,
    model: DiseaseClassifierWithGradCAM,
    N: int = 4,
    P: int = 4,
) -> List[tuple]:
    """
    Returns top-P (score, bbox) tuples from an N×N GradCAM grid.
    FIX [BUG]: image_tensor must be created with requires_grad=True (see caller).
    """
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    patch_h, patch_w = H // N, W // N

    model.zero_grad()
    outputs      = model(image_tensor)
    target_class = torch.argmax(outputs, dim=1).item()
    score        = outputs[0, target_class]
    score.backward(retain_graph=True)

    combined_heatmap = torch.zeros(H, W, device=DEVICE)

    if model.gradients is not None and model.activations is not None:
        pooled_grads = model.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((pooled_grads * model.activations).sum(dim=1))    # (1, h, w)
        combined_heatmap = (
            F.interpolate(cam.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
            .squeeze()
            .detach()
        )
        if combined_heatmap.max() > 1e-6:
            combined_heatmap /= combined_heatmap.max()

    heatmap_cpu = combined_heatmap.cpu().numpy()
    patch_scores: List[tuple] = []

    for i in range(N):
        for j in range(N):
            y1 = i * patch_h;  y2 = (i + 1) * patch_h
            x1 = j * patch_w;  x2 = (j + 1) * patch_w
            patch_scores.append((
                float(heatmap_cpu[y1:y2, x1:x2].mean()),
                (x1, y1, x2, y2),
            ))

    patch_scores.sort(key=lambda t: t[0], reverse=True)
    return patch_scores[:P]


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/users", tags=["User Engine"])
async def create_user(user: UserCreate):
    """Register a new user. Password is hashed before storage."""
    if await users_col.find_one({"email": user.email}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered.")

    # FIX [WARN]: hash password before storing
    hashed_pw = pwd_context.hash(user.password) if user.password else None
    user_id   = str(uuid.uuid4())[:8]
    now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = {
        "user_id"    : user_id,
        "full_name"  : user.full_name,
        "email"      : user.email,
        "password"   : hashed_pw,
        "created_at" : now,
        "total_scans": 0,
        "last_scan_at": None,
    }
    try:
        await users_col.insert_one(doc)
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))

    return {"status": "success", "user_id": user_id,
            "full_name": user.full_name, "created_at": now}


@app.post("/api/login", tags=["User Engine"])
async def login_user(login: UserLogin):
    """Authenticate by email + password."""
    user = await users_col.find_one({"email": login.email})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    # Verify password (allow None password for demo users created without one)
    if user.get("password") and login.password:
        if not pwd_context.verify(login.password, user["password"]):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect password.")

    return {"user_id": user["user_id"], "full_name": user["full_name"],
            "email": user["email"]}


@app.get("/api/users/by-email/{email}", tags=["User Engine"])
async def get_user_by_email(email: str):
    user = await users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    return {"user_id": user["user_id"], "full_name": user["full_name"],
            "email": user["email"], "created_at": user["created_at"],
            "total_scans": user.get("total_scans", 0)}


@app.get("/api/users/{user_id}", tags=["User Engine"])
async def get_user_by_id(user_id: str):
    user = await users_col.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    return {"user_id": user["user_id"], "full_name": user["full_name"],
            "email": user["email"], "created_at": user["created_at"],
            "total_scans": user.get("total_scans", 0),
            "last_scan_at": user.get("last_scan_at")}


@app.post("/api/analyze/{user_id}", tags=["Core Pipeline"])
async def analyze_image(user_id: str, file: UploadFile = File(...)):
    """
    Full scan pipeline:
      MediaPipe face detection → GradCAM patch grid → AE latent → wellness scores.
    ⚠️  DEMO — scores are simulated, not medical advice.
    """
    user = await users_col.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    # ── Load image ────────────────────────────────────────────────────────────
    content  = await file.read()
    nparr    = np.frombuffer(content, np.uint8)
    native_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if native_img is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Could not decode image. Send a valid JPEG/PNG.")

    scan_id   = f"SCAN_{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Persist original ──────────────────────────────────────────────────────
    orig_fname = f"{scan_id}_orig.jpg"
    cv2.imwrite(os.path.join(STORAGE_DIR, orig_fname), native_img)

    # ── Quality checks ────────────────────────────────────────────────────────
    blur_score, brightness_score = run_quality_checks(native_img)
    h_nat, w_nat = native_img.shape[:2]

    # ── Face detection (MediaPipe) ────────────────────────────────────────────
    rgb_img = cv2.cvtColor(native_img, cv2.COLOR_BGR2RGB)
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        det_results = fd.process(rgb_img)

    if det_results.detections:
        bb   = det_results.detections[0].location_data.relative_bounding_box
        xmin = max(0, int(bb.xmin * w_nat))
        ymin = max(0, int(bb.ymin * h_nat))
        xmax = min(w_nat, xmin + int(bb.width  * w_nat))
        ymax = min(h_nat, ymin + int(bb.height * h_nat))
        skin_roi     = native_img[ymin:ymax, xmin:xmax]
        regions_log  = {"face_detected": True,
                        "bounding_box": [xmin, ymin, xmax, ymax]}
    else:
        skin_roi     = native_img.copy()
        regions_log  = {"face_detected": False,
                        "bounding_box": [0, 0, w_nat, h_nat]}

    # Guard against degenerate ROI
    if skin_roi.size == 0:
        skin_roi = native_img.copy()

    skin_roi_224 = cv2.resize(skin_roi,    (224, 224))
    full_224     = cv2.resize(native_img,  (224, 224))

    # ── Body part classification ───────────────────────────────────────────────
    body_tensor = (
        torch.tensor(full_224, dtype=torch.float32)
        .permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    )
    with torch.no_grad():
        body_logits       = body_model(body_tensor)
        detected_body_part = BODY_CLASSES[torch.argmax(body_logits).item()]

    # FIX [BUG]: create input_tensor WITH requires_grad=True so GradCAM hooks fire
    input_tensor = (
        torch.tensor(skin_roi_224, dtype=torch.float32)
        .permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    ).requires_grad_(True)

    # ── GradCAM patch selection ───────────────────────────────────────────────
    top_patches = compute_gradcam_patches(input_tensor, disease_model, N=4, P=4)

    # ── Annotate image ────────────────────────────────────────────────────────
    annotated = skin_roi_224.copy()
    for rank, (intensity, (x1, y1, x2, y2)) in enumerate(top_patches, 1):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"P{rank}", (x1 + 4, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    proc_fname = f"{scan_id}_proc.jpg"
    cv2.imwrite(os.path.join(STORAGE_DIR, proc_fname), annotated)

    # ── Autoencoder latent → wellness scores ──────────────────────────────────
    # FIX [BUG]: safe patch extraction + explicit flatten before indexing
    x1, y1, x2, y2 = top_patches[0][1]
    # Guard: patch region must be non-empty
    patch_region = input_tensor[:, :,
                                max(0, y1):min(224, y2),
                                max(0, x1):min(224, x2)]
    if patch_region.shape[2] < 2 or patch_region.shape[3] < 2:
        patch_region = input_tensor  # fallback to full image

    ae_in = F.interpolate(patch_region.detach(), size=(56, 56),
                          mode="bilinear", align_corners=False)

    with torch.no_grad():
        latent_vec, _ = ae_model(ae_in)

    # FIX [BUG]: explicit flatten to guarantee 1-D indexing
    v = latent_vec.squeeze().flatten().cpu().numpy()  # shape (8,)
    stress_index    = int(abs(v[0]) * 100) % 100
    fatigue_index   = int(abs(v[1]) * 100) % 100
    hydration_level = int(100 - (abs(v[2]) * 100) % 100)
    overall_score   = int((hydration_level + (100 - stress_index) +
                           (100 - fatigue_index)) // 3)

    recommendations = [
        "Practice 4-7-8 deep breathing daily to lower cortisol levels.",
        "Prioritise 7-9 hours of sleep and use cooling eye masks to reduce puffiness.",
        "Drink at least 2.5 L of water daily; add electrolytes during exercise.",
    ]

    # ── Persist to MongoDB ────────────────────────────────────────────────────
    scan_doc = {
        "scan_id"              : scan_id,
        "user_id"              : user_id,
        "user_email"           : user["email"],
        "user_name"            : user["full_name"],
        "original_image_path"  : f"/static/{orig_fname}",
        "processed_image_path" : f"/static/{proc_fname}",
        "detected_body_part"   : detected_body_part,
        "blur_score"           : blur_score,
        "brightness_score"     : brightness_score,
        "stress_index"         : stress_index,
        "fatigue_index"        : fatigue_index,
        "hydration_level"      : hydration_level,
        "overall_wellness_score": overall_score,
        "recommendations"      : recommendations,
        "timestamp"            : timestamp,
        "quality_status"       : "Acceptable" if blur_score > 50 else "Suboptimal",
    }
    await scans_col.insert_one(scan_doc)
    await users_col.update_one(
        {"user_id": user_id},
        {"$inc": {"total_scans": 1}, "$set": {"last_scan_at": timestamp}},
    )

    return {
        "status"    : "success",
        "scan_id"   : scan_id,
        "user_id"   : user_id,
        "timestamp" : timestamp,
        "disclaimer": "⚠️ DEMO PROTOTYPE: wellness metrics are simulated — not medical diagnoses.",
        "dashboard_data": {
            "image_previews": {
                "original_url" : f"/static/{orig_fname}",
                "processed_url": f"/static/{proc_fname}",
            },
            "segmentation": {
                "detected_body_region": detected_body_part,
                "structural_metadata" : regions_log,
            },
            "quality_metrics": {
                "sharpness_index"  : blur_score,
                "brightness_index" : brightness_score,
                "status"           : "Acceptable" if blur_score > 50 else "Suboptimal",
            },
            "wellness_biometrics": {
                "stress_index"           : stress_index,
                "fatigue_index"          : fatigue_index,
                "hydration_level"        : hydration_level,
                "composite_wellness_score": overall_score,
            },
            "actionable_interventions": recommendations,
        },
    }


@app.get("/api/history/{user_id}", tags=["Core Pipeline"])
async def get_scan_history(user_id: str):
    cursor = scans_col.find({"user_id": user_id}).sort("timestamp", -1)
    scans  = await cursor.to_list(length=None)
    return {
        "user_id"    : user_id,
        "total_scans": len(scans),
        "history"    : [
            {
                "scan_id"       : s["scan_id"],
                "timestamp"     : s["timestamp"],
                "body_part"     : s["detected_body_part"],
                "wellness_score": s["overall_wellness_score"],
                "biometrics"    : {"stress": s["stress_index"],
                                   "fatigue": s["fatigue_index"],
                                   "hydration": s["hydration_level"]},
                "quality"       : {"blur": s["blur_score"],
                                   "brightness": s["brightness_score"]},
                "urls"          : {"original" : s["original_image_path"],
                                   "processed": s["processed_image_path"]},
                "recommendations": s["recommendations"],
            }
            for s in scans
        ],
    }


@app.get("/api/admin/records", tags=["Admin Portal"])
async def admin_all_records():
    cursor = scans_col.find().sort("timestamp", -1).limit(100)
    scans  = await cursor.to_list(length=None)
    return {
        "admin_audit_log_records": [
            {
                "scan_id"     : s["scan_id"],
                "timestamp"   : s["timestamp"],
                "user_details": {"id"   : s["user_id"],
                                 "name" : s.get("user_name",  "Unknown"),
                                 "email": s.get("user_email", "Unknown")},
                "vision_output": {"body_part" : s["detected_body_part"],
                                  "blur"      : s["blur_score"],
                                  "brightness": s["brightness_score"]},
                "biometric_scores": {"stress"   : s["stress_index"],
                                     "fatigue"  : s["fatigue_index"],
                                     "hydration": s["hydration_level"],
                                     "composite": s["overall_wellness_score"]},
                "artifacts"  : {"orig_url": s["original_image_path"],
                                "proc_url": s["processed_image_path"]},
            }
            for s in scans
        ]
    }


@app.get("/api/admin/summary", tags=["Admin Portal"])
async def admin_summary():
    total_users = await users_col.count_documents({})
    total_scans = await scans_col.count_documents({})

    agg = await scans_col.aggregate([
        {"$group": {"_id": None, "avg_wellness": {"$avg": "$overall_wellness_score"}}}
    ]).to_list(1)
    avg_wellness = round(agg[0]["avg_wellness"], 1) if agg else 0.0

    daily = await scans_col.aggregate([
        {"$group": {"_id": {"$substr": ["$timestamp", 0, 10]}, "count": {"$sum": 1}}},
        {"$sort": {"_id": -1}},
        {"$limit": 7},
    ]).to_list(7)

    return {
        "system_status"                  : "ONLINE",
        "database"                       : "MongoDB Atlas",
        "total_registered_users"         : total_users,
        "total_scans_processed"          : total_scans,
        "average_population_wellness_score": avg_wellness,
        "daily_scan_trends"              : [{"date": d["_id"], "count": d["count"]}
                                            for d in daily],
    }


@app.delete("/api/users/{user_id}", tags=["Admin Portal"])
async def delete_user(user_id: str):
    u_res = await users_col.delete_one({"user_id": user_id})
    if u_res.deleted_count == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    s_res = await scans_col.delete_many({"user_id": user_id})
    return {"status": "success",
            "message": f"Deleted user {user_id} and {s_res.deleted_count} scan(s)."}


@app.get("/api/health", tags=["System"])
async def health_check():
    try:
        await mongo_client.admin.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return {
        "status"                : "healthy",
        "database_connectivity" : db_status,
        "neural_networks_active": body_model is not None,
        "device"                : str(DEVICE),
        "timestamp"             : datetime.now().isoformat(),
    }


# FIX [BUG]: correct module:app string (was "app.py:app")
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
