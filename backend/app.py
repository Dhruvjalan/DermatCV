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

# ─── Load Configuration ──────────────────────────────────────────────────────
load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MONGODB_URI   = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "wellness_analytics")

_raw_origins  = os.getenv("CORS_ORIGINS", "http://localhost:3000")
CORS_ORIGINS  = [o.strip() for o in _raw_origins.split(",")]

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
STORAGE_DIR    = os.getenv("STORAGE_DIR",    "storage_vault")

os.makedirs(STORAGE_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Global State Vectors ────────────────────────────────────────────────────
BODY_CLASSES = ["Face", "Skin Hand", "Eye", "Forehead", "Cheek", "Neck", "Arm", "Leg", "Chest", "Back"]
DISEASE_CLASSES = [
        "Redness",
        "dark spots",
        "inflammatory acne",
        "non inflammatory acne black heads",
        "non inflammatory acne white heads",
        "pigmentation",
        "pores",
        "wrinkles"
    ]

body_model = None
disease_model = None
ae_model = None
db = None
users_col = None
scans_col = None
mongo_client = None

# Simple in-memory Knowledge Base for our RAG Pipeline
WELLNESS_KNOWLEDGE_BASE = [
    {"condition": "Surface Fatigue", "text": "Surface fatigue is marked by micro-vessel restriction. Interventions require cooling therapy masks, 7-9 hours of structured circadian sleep, and topical adaptogens like Green Tea Extract or Vitamin C to clear oxidative layout stress."},
    {"condition": "Dehydration Zone", "text": "Dehydration zones feature compromised lipid barrier metrics. Protocols demand immediate replenishment of 2.5 to 3 Liters of mineralized water daily, atmospheric humidification, and topical hyaluronic moisture binding agent application."},
    {"condition": "Vascular Flush", "text": "Vascular Flush indicates elevated cutaneous thermal load and inflammation. Mitigate via vagus nerve down-regulation, 4-7-8 deep breathing techniques to check systemic cortisol spikes, and elimination of vasodilating trigger compounds."},
    {"condition": "Healthy Base", "text": "Healthy base maintenance relies on preventative stabilization. Preserve with low-glycemic metabolic intake, antioxidant support structures, and baseline aerobic physical movement to support cellular microcirculation."}
]

# ─── Deep Learning Architectures ──────────────────────────────────────────────
class BodyPartClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56
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
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(64 * 28 * 28, num_classes)
        self.gradients   : Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def _activations_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 112
        x = self.pool(F.relu(self.conv2(x)))   # 56
        x = F.relu(self.conv3(x))              # 56
        self.activations = x
        if x.requires_grad:
            x.register_hook(self._activations_hook)
        x = self.pool(x)                        # 28
        return self.fc(x.view(x.size(0), -1))


class NormalAutoencoder(nn.Module):
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


# ─── Integrated LLM Engine & RAG Retrieval Routing ────────────────────────────
class RagRecommendationEngine:
    """
    Simulates a localized semantic vector retrieval matching model and pipelines context
    into the hosted Enterprise LLM completion block (OpenAI/Ollama wrapper alternative).
    """
    @staticmethod
    def get_embedding_mock(text: str) -> np.ndarray:
        # Fast, predictable projection for deterministic semantic matching
        hash_val = sum(ord(c) for c in text)
        np.random.seed(hash_val % 1000)
        vec = np.random.randn(64)
        return vec / np.linalg.norm(vec)

    @classmethod
    def retrieve_context(cls, detected_condition: str, top_k: int = 1) -> str:
        query_vec = cls.get_embedding_mock(detected_condition)
        scored_contexts = []
        
        for doc in WELLNESS_KNOWLEDGE_BASE:
            doc_vec = cls.get_embedding_mock(doc["text"])
            similarity = float(np.dot(query_vec, doc_vec))
            # Hard boost context if keywords explicitly map up
            if doc["condition"].lower() in detected_condition.lower():
                similarity += 2.0
            scored_contexts.append((similarity, doc["text"]))
            
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        return " ".join([item[1] for item in scored_contexts[:top_k]])

    @classmethod
    def generate_personalized_interventions(cls, condition: str, stress: int, fatigue: int, hydration: int) -> List[str]:
        context_document = cls.retrieve_context(condition)
        
        # Real-world system call mapping block mock representing OpenAI / Anthropic SDK processing:
        # response = openai.ChatCompletion.create(messages=[{"role": "user", "content": ...}])
        
        llm_prompt = (
            f"SYSTEM: You are a clinical wellness generator. Context: {context_document}. "
            f"Format exactly 3 direct, short bullet items tailored to: {condition} "
            f"with Stress={stress}%, Fatigue={fatigue}%, Hydration={hydration}%."
        )
        
        # Generated downstream output matching current metrics state dynamically
        if "Dehydration" in condition or hydration < 50:
            return [
                f"Escalate raw hydration targets. Given your low {hydration}% index, ingest 3.2L electrolyte fluids daily.",
                "Deploy structural barrier locked topical serums containing pure low-molecular weight hyaluronic chains.",
                "Halt all oxidative cardiovascular exertion cycles until cellular fluid volumes recover safely."
            ]
        elif "Fatigue" in condition or fatigue > 60:
            return [
                f"Prioritize immediate neural recovery loops to combat your high {fatigue}% fatigue signature.",
                "Incorporate advanced cold-exposure protocols for 120 seconds post-waking to spark systemic circulation.",
                "Enforce strict blue-light mitigation fields across all digital surfaces starting 90 minutes before sleep."
            ]
        elif "Flush" in condition or stress > 60:
            return [
                "Execute 3 cycles of localized thermal cooling compress maps over high reactive dermal clusters.",
                f"Engage the parasympathetic response system with 5 minutes of 4-7-8 breathing to target the {stress}% stress spike.",
                "Eliminate active micro-inflammatory dietary inputs and high vasodilation culinary ingredients."
            ]
        else:
            return [
                "Preserve current stabilization values by maintaining a strict baseline of micro-nutrient profile loading.",
                "Engage in active mobility sequences to boost functional peripheral vascular flow fields.",
                "Conduct standard weekly validation analysis sweeps to optimize your positive cellular baseline."
            ]


# ─── Lifespan Architecture Manager ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: "FastAPI"):
    global body_model, disease_model, ae_model
    global db, users_col, scans_col, mongo_client, BODY_CLASSES, DISEASE_CLASSES

    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is missing. Verify your ecosystem environment layout variables.")

    print("🔗 Initializing MongoDB Connection Pool...")
    mongo_client = AsyncIOMotorClient(
        MONGODB_URI,
        maxPoolSize=50,
        minPoolSize=10,
        maxIdleTimeMS=60000,
    )
    db = mongo_client[DATABASE_NAME]
    users_col = db["users"]
    scans_col = db["scans"]

    # Generate indices
    await users_col.create_index("email", unique=True, background=True)
    await users_col.create_index("user_id", unique=True, background=True)
    await scans_col.create_index("scan_id", unique=True, background=True)
    await scans_col.create_index("user_id", background=True)

    # ─── FIX: Instantiate models to match checkpoint shapes ───────────
    # body_classifier.pth expects 10 outputs. disease_classifier.pth expects 8.
    body_model    = BodyPartClassifier(num_classes=10).to(DEVICE)
    disease_model = DiseaseClassifierWithGradCAM(num_classes=8).to(DEVICE)
    ae_model      = NormalAutoencoder().to(DEVICE)

    # System Checkpoint Hydrator
    for fname, target_m in [
        ("body_classifier.pth", body_model),
        ("disease_classifier.pth", disease_model),
        ("patch_autoencoder.pth", ae_model)
    ]:
        path = os.path.join(CHECKPOINT_DIR, fname)
        if os.path.exists(path):
            target_m.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"  ✅ Hydrated: {fname}")
            
            # ─── FIX: Dynamically re-map output layers back to code settings ───
            if fname == "body_classifier.pth" and len(BODY_CLASSES) != 10:
                print(f"  🔧 Adjusting BodyPartClassifier output from 10 down to {len(BODY_CLASSES)} channels...")
                body_model.classifier[3] = nn.Linear(128, len(BODY_CLASSES)).to(DEVICE)
                
            if fname == "disease_classifier.pth" and len(DISEASE_CLASSES) != 8:
                print(f"  🔧 Adjusting DiseaseClassifier output from 8 down to {len(DISEASE_CLASSES)} channels...")
                disease_model.fc = nn.Linear(64 * 28 * 28, len(DISEASE_CLASSES)).to(DEVICE)
        else:
            print(f"  ⚠️ Checkpoint {fname} missing — operating inside standard random simulation space.")

    body_model.eval()
    disease_model.eval()
    ae_model.eval()
    yield
    if mongo_client:
        mongo_client.close()
# ─── Server Application Interface ─────────────────────────────────────────────
app = FastAPI(
    title="BioCV Enterprise Wellness Analytics Core",
    version="2.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")

class UserCreate(BaseModel):
    full_name: str
    email: str
    password: Optional[str] = None

class UserLogin(BaseModel):
    email   : str
    password: Optional[str] = None

# Helper fn


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

# def run_quality_checks(img: np.ndarray):
#     gray             = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur_score       = int(cv2.Laplacian(gray, cv2.CV_64F).var())
#     lab              = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     brightness_score = int(np.mean(cv2.split(lab)[0]))
#     return blur_score, brightness_score



# api endoints
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




# ─── Optimized Core CV Pipeline Inference Engine ────────────────────────────────
@app.post("/api/analyze/{user_id}", tags=["Core Pipeline"])
async def analyze_image(user_id: str, file: UploadFile = File(...)):
    user = await users_col.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Target profile user identification token verified as absent.")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    native_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if native_img is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Image byte processing extraction exception. Send valid matrix stream.")

    scan_id = f"SCAN_{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Image Persistence
    orig_fname = f"{scan_id}_orig.jpg"
    cv2.imwrite(os.path.join(STORAGE_DIR, orig_fname), native_img)

    blur_score, brightness_score = run_quality_checks(native_img)
    h_nat, w_nat = native_img.shape[:2]

    # MediaPipe Face Extractor
    rgb_img = cv2.cvtColor(native_img, cv2.COLOR_BGR2RGB)
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        det_results = fd.process(rgb_img)

    if det_results.detections:
        bb = det_results.detections[0].location_data.relative_bounding_box
        xmin = max(0, int(bb.xmin * w_nat))
        ymin = max(0, int(bb.ymin * h_nat))
        xmax = min(w_nat, xmin + int(bb.width * w_nat))
        ymax = min(h_nat, ymin + int(bb.height * h_nat))
        skin_roi = native_img[ymin:ymax, xmin:xmax]
        regions_log = {"face_detected": True, "bounding_box": [xmin, ymin, xmax, ymax]}
    else:
        skin_roi = native_img.copy()
        regions_log = {"face_detected": False, "bounding_box": [0, 0, w_nat, h_nat]}

    if skin_roi.size == 0:
        skin_roi = native_img.copy()

    skin_roi_224 = cv2.resize(skin_roi, (224, 224))
    full_224 = cv2.resize(native_img, (224, 224))

    # 1. Body Part Classification Inference
    body_tensor = torch.tensor(full_224, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        body_logits = body_model(body_tensor)
        detected_body_part = BODY_CLASSES[torch.argmax(body_logits).item()]

    # 2. Disease Condition Classification Inference & Hook Initialization
    input_tensor = (torch.tensor(skin_roi_224, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0).requires_grad_(True)
    
    # Run the core inference classification pass to fix structural runtime vacancy
    disease_logits = disease_model(input_tensor)
    detected_class_index = torch.argmax(disease_logits, dim=1).item()
    detected_disease_condition = DISEASE_CLASSES[detected_class_index]

    # Compute GradCAM based on actual forward engine activations
    top_patches = compute_gradcam_patches(input_tensor, disease_model, N=4, P=4)

    # Output Annotations mapping
    annotated = skin_roi_224.copy()
    for rank, (intensity, (x1, y1, x2, y2)) in enumerate(top_patches, 1):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"P{rank}", (x1 + 4, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    proc_fname = f"{scan_id}_proc.jpg"
    cv2.imwrite(os.path.join(STORAGE_DIR, proc_fname), annotated)

    # 3. Patch Autoencoder Auto-Evaluation Tracking
    x1, y1, x2, y2 = top_patches[0][1]
    patch_region = input_tensor[:, :, max(0, y1):min(224, y2), max(0, x1):min(224, x2)]
    if patch_region.shape[2] < 2 or patch_region.shape[3] < 2:
        patch_region = input_tensor

    ae_in = F.interpolate(patch_region.detach(), size=(56, 56), mode="bilinear", align_corners=False)
    with torch.no_grad():
        latent_vec, _ = ae_model(ae_in)

    v = latent_vec.squeeze().flatten().cpu().numpy()
    stress_index    = int(abs(v[0]) * 100) % 100
    fatigue_index   = int(abs(v[1]) * 100) % 100
    hydration_level = int(100 - (abs(v[2]) * 100) % 100)
    overall_score   = int((hydration_level + (100 - stress_index) + (100 - fatigue_index)) // 3)

    # 4. Neural RAG Execution Core
    recommendations = RagRecommendationEngine.generate_personalized_interventions(
        condition=detected_disease_condition,
        stress=stress_index,
        fatigue=fatigue_index,
        hydration=hydration_level
    )

    # Document Mapping to Database Cluster Collections
    scan_doc = {
        "scan_id": scan_id,
        "user_id": user_id,
        "user_email": user["email"],
        "user_name": user["full_name"],
        "original_image_path": f"/static/{orig_fname}",
        "processed_image_path": f"/static/{proc_fname}",
        "detected_body_part": detected_body_part,
        "detected_condition": detected_disease_condition, # Saved inference type value mapping
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "stress_index": stress_index,
        "fatigue_index": fatigue_index,
        "hydration_level": hydration_level,
        "overall_wellness_score": overall_score,
        "recommendations": recommendations,
        "timestamp": timestamp,
        "quality_status": "Acceptable" if blur_score > 50 else "Suboptimal",
    }
    
    await scans_col.insert_one(scan_doc)
    await users_col.update_one(
        {"user_id": user_id},
        {"$inc": {"total_scans": 1}, "$set": {"last_scan_at": timestamp}},
    )

    return {
        "status": "success",
        "scan_id": scan_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "dashboard_data": {
            "image_previews": {
                "original_url": f"/static/{orig_fname}",
                "processed_url": f"/static/{proc_fname}",
            },
            "segmentation": {
                "detected_body_region": detected_body_part,
                "detected_condition_inference": detected_disease_condition,
                "structural_metadata": regions_log,
            },
            "quality_metrics": {
                "sharpness_index": blur_score,
                "brightness_index": brightness_score,
                "status": "Acceptable" if blur_score > 50 else "Suboptimal",
            },
            "wellness_biometrics": {
                "stress_index": stress_index,
                "fatigue_index": fatigue_index,
                "hydration_level": hydration_level,
                "composite_wellness_score": overall_score,
            },
            "actionable_interventions": recommendations,
        },
    }

@app.get("/api/history/{user_id}", tags=["Core Pipeline"])
async def get_scan_history(user_id: str):
    cursor = scans_col.find({"user_id": user_id}).sort("timestamp", -1)
    scans = await cursor.to_list(length=None)
    return {
        "user_id": user_id,
        "total_scans": len(scans),
        "history": [
            {
                "scan_id": s["scan_id"],
                "timestamp": s["timestamp"],
                "body_part": s["detected_body_part"],
                "condition": s.get("detected_condition", "Evaluation Base"),
                "wellness_score": s["overall_wellness_score"],
                "biometrics": {"stress": s["stress_index"], "fatigue": s["fatigue_index"], "hydration": s["hydration_level"]},
                "quality": {"blur": s["blur_score"], "brightness": s["brightness_score"]},
                "urls": {"original": s["original_image_path"], "processed": s["processed_image_path"]},
                "recommendations": s["recommendations"],
            }
            for s in scans
        ],
    }

# Fallback wrapper utility for internal execution 

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



def run_quality_checks(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = int(cv2.Laplacian(gray, cv2.CV_64F).var())
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    brightness_score = int(np.mean(cv2.split(lab)[0]))
    return blur_score, brightness_score

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)