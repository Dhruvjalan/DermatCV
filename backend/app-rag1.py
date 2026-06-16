import os
import json
import uuid
import cv2
import re
import numpy as np
import networkx as nx
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b3
import mediapipe as mp
from PIL import Image
from groq import Groq

# ── ENVIRONMENT & CONFIG SETUP ──────────────────────────────────────────

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

MONGODB_URI   = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "wellness_analytics")

_raw_origins  = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
CORS_ORIGINS  = [o.strip() for o in _raw_origins.split(",")]

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
STORAGE_DIR    = os.getenv("STORAGE_DIR",    "storage_vault")

os.makedirs(STORAGE_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── GROQ & GRAPH CONFIG ─────────────────────────────────────────────────

BASE_DIR = os.path.join(os.path.dirname(__file__), "disease_graph")
SESSION_DIR = os.path.join(BASE_DIR, "sessions")
GRAPH_GRAPHML_PATH = os.path.join(BASE_DIR, "disease_symptom_graph.graphml")
GRAPH_HTML_PATH = os.path.join(BASE_DIR, "graph_interactive.html")
GRAPH_STATIC_PATH = os.path.join(BASE_DIR, "graph_static.png")

Path(SESSION_DIR).mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "llama-3.3-70b-versatile"
SMALL_MODEL = "llama-3.1-8b-instant"

if os.path.exists(GRAPH_GRAPHML_PATH):
    G = nx.read_graphml(GRAPH_GRAPHML_PATH)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    print(f"Loaded Graph: {G.number_of_nodes()} nodes.")
else:
    G = nx.DiGraph()
    print("Warning: Graph file not found. Created empty graph.")

# ── ML MODEL DEFINITIONS ────────────────────────────────────────────────

BODY_CLASSES = ["Face", "Skin Hand", "Eye", "Forehead", "Cheek", "Neck", "Arm", "Leg", "Chest", "Back"]

# Updated to match the exactly 10 classes from your CV_Model_Training.ipynb
DISEASE_CLASSES = [
    'Redness', 'acne', 'blackheades', 'dark spots', 'inflammatory acne', 
    'non inflammatory acne black heads', 'non inflammatory acne white heads', 
    'pigmentation', 'pores', 'wrinkles'
]

class BodyPartClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).view(x.size(0), -1))

# Replaces the mock CNN with your EfficientNet-B3 architecture from the Notebook
class SkinDiseaseModelWithGradCAM(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = efficientnet_b3(weights=None)
        
        self.features = base.features
        self.avgpool = base.avgpool
        
        # Exact classifier head from notebook
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(1536, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        self.gradients : Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def _activations_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        self.activations = x
        if x.requires_grad:
            x.register_hook(self._activations_hook)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class NormalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=2, padding=1), nn.ReLU(),   
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   
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

# Transform required by EfficientNet-B3 (Notebook pipeline)
notebook_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── RAG ENGINE (GROQ DRIVEN) ────────────────────────────────────────────

WELLNESS_KNOWLEDGE_BASE = [
    {"condition": "Surface Fatigue", "text": "Surface fatigue is marked by micro-vessel restriction. Interventions require cooling therapy masks, 7-9 hours of structured circadian sleep, and topical adaptogens like Green Tea Extract or Vitamin C to clear oxidative layout stress."},
    {"condition": "Dehydration Zone", "text": "Dehydration zones feature compromised lipid barrier metrics. Protocols demand immediate replenishment of 2.5 to 3 Liters of mineralized water daily, atmospheric humidification, and topical hyaluronic moisture binding agent application."},
    {"condition": "Vascular Flush", "text": "Vascular Flush indicates elevated cutaneous thermal load and inflammation. Mitigate via vagus nerve down-regulation, 4-7-8 deep breathing techniques to check systemic cortisol spikes, and elimination of vasodilating trigger compounds."},
    {"condition": "Healthy Base", "text": "Healthy base maintenance relies on preventative stabilization. Preserve with low-glycemic metabolic intake, antioxidant support structures, and baseline aerobic physical movement to support cellular microcirculation."}
]

class RagRecommendationEngine:
    @classmethod
    def retrieve_context(cls, detected_condition: str) -> str:
        # Simplified string-matching retrieval (expand to vector-db later if desired)
        matched = [doc["text"] for doc in WELLNESS_KNOWLEDGE_BASE if doc["condition"].lower() in detected_condition.lower()]
        return matched[0] if matched else WELLNESS_KNOWLEDGE_BASE[-1]["text"]

    @classmethod
    def generate_personalized_interventions(cls, condition: str, stress: int, fatigue: int, hydration: int) -> List[str]:
        context_document = cls.retrieve_context(condition)
        
        prompt = (
            f"CLINICAL CONTEXT: {context_document}\n\n"
            f"PATIENT DATA:\n- Detected Condition: {condition}\n- Stress Level: {stress}%\n"
            f"- Fatigue Level: {fatigue}%\n- Hydration Index: {hydration}%\n\n"
            "TASK: Based on the clinical context and patient data, generate exactly 3 direct, short, highly actionable "
            "clinical wellness bullet points tailored specifically to this user's current biometrics. "
            "Format as a plain list without numbers or markdown bullets."
        )
        
        try:
            resp = groq_client.chat.completions.create(
                model=SMALL_MODEL,
                messages=[
                    {"role": "system", "content": "You are an elite clinical dermatologist and wellness AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=200
            )
            response_text = resp.choices[0].message.content.strip()
            # Clean up the output into a list
            bullets = [line.strip().lstrip('-').lstrip('*').strip() for line in response_text.split('\n') if line.strip()]
            return bullets[:3] if len(bullets) >= 3 else bullets
            
        except Exception as e:
            print(f"Groq API Error in RAG: {e}")
            # Fallback
            return [
                f"Implement specific protocols tailored to your {condition}.",
                f"Address your {stress}% stress baseline through down-regulation.",
                f"Maintain deep hydration frameworks considering your {hydration}% fluid index."
            ]


# ── LIFESPAN & FASTAPI INITIALIZATION ───────────────────────────────────

body_model = None
disease_model = None
ae_model = None
db = None
users_col = None
scans_col = None
mongo_client = None

@asynccontextmanager
async def lifespan(application: "FastAPI"):
    global body_model, disease_model, ae_model
    global db, users_col, scans_col, mongo_client

    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is missing.")

    print("🔗 Initializing MongoDB Connection Pool...")
    mongo_client = AsyncIOMotorClient(
        MONGODB_URI, maxPoolSize=50, minPoolSize=10, maxIdleTimeMS=60000,
    )
    db = mongo_client[DATABASE_NAME]
    users_col = db["users"]
    scans_col = db["scans"]

    await users_col.create_index("email", unique=True, background=True)
    await users_col.create_index("user_id", unique=True, background=True)
    await scans_col.create_index("scan_id", unique=True, background=True)
    await scans_col.create_index("user_id", background=True)

    body_model    = BodyPartClassifier(num_classes=10).to(DEVICE)
    disease_model = SkinDiseaseModelWithGradCAM(num_classes=len(DISEASE_CLASSES)).to(DEVICE)
    ae_model      = NormalAutoencoder().to(DEVICE)

    for fname, target_m in [
        ("body_classifier.pth", body_model),
        ("efficientnet_best.pt", disease_model), # Updated to load notebook's model
        ("patch_autoencoder.pth", ae_model)
    ]:
        path = os.path.join(CHECKPOINT_DIR, fname)
        if os.path.exists(path):
            target_m.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"  ✅ Hydrated: {fname}")
        else:
            print(f"  ⚠️ Checkpoint {fname} missing — operating inside standard random simulation space.")

    body_model.eval()
    disease_model.eval()
    ae_model.eval()
    yield
    if mongo_client:
        mongo_client.close()

app = FastAPI(
    title="DermatCV Enterprise Wellness Analytics Core",
    version="2.3.0",
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


# ── UTILS ───────────────────────────────────────────────────────────────

def run_quality_checks(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = int(cv2.Laplacian(gray, cv2.CV_64F).var())
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    brightness_score = int(np.mean(cv2.split(lab)[0]))
    return blur_score, brightness_score

def compute_gradcam_patches(
    image_tensor: torch.Tensor,
    model: SkinDiseaseModelWithGradCAM,
    N: int = 4,
    P: int = 4,
) -> List[tuple]:
    """Returns top-P (score, bbox) tuples from an N×N GradCAM grid."""
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    patch_h, patch_w = H // N, W // N

    model.zero_grad()
    outputs      = model(image_tensor)
    target_class = torch.argmax(outputs, dim=1).item()
    score        = outputs[0, target_class]
    score.backward(retain_graph=True)

    combined_heatmap = torch.zeros(H, W, device=DEVICE)

    if model.gradients is not None and model.activations is not None:
        pooled_grads = model.gradients.mean(dim=[2, 3], keepdim=True)  
        cam = F.relu((pooled_grads * model.activations).sum(dim=1))    
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


# ── SCHEMAS ─────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    full_name: str
    email: str
    password: Optional[str] = None

class UserLogin(BaseModel):
    email   : str
    password: Optional[str] = None

class HyperParams(BaseModel):
    max_questions: int = Field(default=6, description="Hard dropout after this many Q&A turns")
    confidence_threshold: float = Field(default=3.5, description="Early-exit ratio")
    top_k_candidates: int = Field(default=8, description="Diseases to track at once")
    min_score_to_keep: float = Field(default=0.01, description="Prune below this fraction")
    confirmed_weight_mult: float = Field(default=2.0, description="Multiplier when symptom confirmed")
    denied_weight_mult: float = Field(default=0.15, description="Multiplier when symptom denied")

class InitialDiagnosticRequest(BaseModel):
    user_id: Optional[str] = None  # Added to fetch ML scan history
    symptom_text: str
    hyperparams: Optional[HyperParams] = HyperParams()

class AnswerRequest(BaseModel):
    session_id: str
    answer: str

class UpdateGraphRequest(BaseModel):
    data: List[Dict[str, str]]


# ── CORE ROUTES ─────────────────────────────────────────────────────────
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
                "biometrics": {
                    "stress": s["stress_index"], 
                    "fatigue": s["fatigue_index"], 
                    "hydration": s["hydration_level"]
                },
                "quality": {
                    "blur": s["blur_score"], 
                    "brightness": s["brightness_score"]
                },
                "urls": {
                    "original": s["original_image_path"], 
                    "processed": s["processed_image_path"]
                },
                "recommendations": s["recommendations"],
            }
            for s in scans
        ],
    }

# Make sure this block remains at the absolute bottom of app.py
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

@app.post("/api/users", tags=["User Engine"])
async def create_user(user: UserCreate):
    if await users_col.find_one({"email": user.email}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered.")
    hashed_pw = pwd_context.hash(user.password) if user.password else None
    user_id   = str(uuid.uuid4())[:8]
    now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = {"user_id": user_id, "full_name": user.full_name, "email": user.email, "password": hashed_pw, "created_at": now, "total_scans": 0, "last_scan_at": None}
    await users_col.insert_one(doc)
    return {"status": "success", "user_id": user_id, "full_name": user.full_name, "created_at": now}

@app.post("/api/login", tags=["User Engine"])
async def login_user(login: UserLogin):
    user = await users_col.find_one({"email": login.email})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    if user.get("password") and login.password:
        if not pwd_context.verify(login.password, user["password"]):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect password.")
    return {"user_id": user["user_id"], "full_name": user["full_name"], "email": user["email"]}

@app.get("/api/users/{user_id}", tags=["User Engine"])
async def get_user_by_id(user_id: str):
    user = await users_col.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    user["_id"] = str(user["_id"])
    return user

@app.post("/api/analyze/{user_id}", tags=["Core Pipeline"])
async def analyze_image(user_id: str, file: UploadFile = File(...)):
    user = await users_col.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User identification token absent.")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    native_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if native_img is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Image byte processing exception.")

    scan_id = f"SCAN_{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    orig_fname = f"{scan_id}_orig.jpg"
    cv2.imwrite(os.path.join(STORAGE_DIR, orig_fname), native_img)

    blur_score, brightness_score = run_quality_checks(native_img)
    h_nat, w_nat = native_img.shape[:2]

    # Face Detection Pipeline
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

    # 1. Body Part Inference (Simple resize for mock architecture)
    full_224 = cv2.resize(native_img, (224, 224))
    body_tensor = torch.tensor(full_224, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        body_logits = body_model(body_tensor)
        detected_body_part = BODY_CLASSES[torch.argmax(body_logits).item()]

    # 2. Disease Classification & GradCAM (Notebook Architecture Integration)
    # Applying the precise standardizations established in the notebook
    pil_roi = Image.fromarray(cv2.cvtColor(skin_roi, cv2.COLOR_BGR2RGB))
    input_tensor = notebook_transform(pil_roi).unsqueeze(0).to(DEVICE).requires_grad_(True)
    
    disease_logits = disease_model(input_tensor)
    detected_class_index = torch.argmax(disease_logits, dim=1).item()
    detected_disease_condition = DISEASE_CLASSES[detected_class_index]

    # GradCAM patch generation on the notebook architecture
    top_patches = compute_gradcam_patches(input_tensor, disease_model, N=4, P=4)

    annotated = cv2.resize(skin_roi, (224, 224)).copy()
    for rank, (intensity, (x1, y1, x2, y2)) in enumerate(top_patches, 1):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"P{rank}", (x1 + 4, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    proc_fname = f"{scan_id}_proc.jpg"
    cv2.imwrite(os.path.join(STORAGE_DIR, proc_fname), annotated)

    # 3. Patch Autoencoder extraction
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

    # 4. Generative RAG Inference
    recommendations = RagRecommendationEngine.generate_personalized_interventions(
        condition=detected_disease_condition,
        stress=stress_index,
        fatigue=fatigue_index,
        hydration=hydration_level
    )

    scan_doc = {
        "scan_id": scan_id, "user_id": user_id, "user_email": user["email"], "user_name": user["full_name"],
        "original_image_path": f"/static/{orig_fname}", "processed_image_path": f"/static/{proc_fname}",
        "detected_body_part": detected_body_part, "detected_condition": detected_disease_condition, 
        "blur_score": blur_score, "brightness_score": brightness_score,
        "stress_index": stress_index, "fatigue_index": fatigue_index, "hydration_level": hydration_level,
        "overall_wellness_score": overall_score, "recommendations": recommendations,
        "timestamp": timestamp, "quality_status": "Acceptable" if blur_score > 50 else "Suboptimal",
    }
    
    await scans_col.insert_one(scan_doc)
    await users_col.update_one({"user_id": user_id}, {"$inc": {"total_scans": 1}, "$set": {"last_scan_at": timestamp}})

    return {
        "status": "success", "scan_id": scan_id, "user_id": user_id, "timestamp": timestamp,
        "dashboard_data": {
            "image_previews": {"original_url": f"/static/{orig_fname}", "processed_url": f"/static/{proc_fname}"},
            "segmentation": {"detected_body_region": detected_body_part, "detected_condition_inference": detected_disease_condition, "structural_metadata": regions_log},
            "quality_metrics": {"sharpness_index": blur_score, "brightness_index": brightness_score, "status": "Acceptable" if blur_score > 50 else "Suboptimal"},
            "wellness_biometrics": {"stress_index": stress_index, "fatigue_index": fatigue_index, "hydration_level": hydration_level, "composite_wellness_score": overall_score},
            "actionable_interventions": recommendations,
        },
    }

# ── DIAGNOSTIC GRAPH / RL ROUTES ────────────────────────────────────────

class DiagnosticSession:
    def __init__(self, session_id: str, ml_predictions: list, hyperparams: dict, user_query: str = "", scan_context: str = ""):
        self.session_id = session_id
        self.started_at = datetime.utcnow().isoformat() + "Z"
        self.ml_predictions = ml_predictions
        self.hyperparams = hyperparams
        self.user_query = user_query          # The user's initial doubt
        self.scan_context = scan_context      # The ML output from MongoDB
        self.confirmed_symptoms = []
        self.denied_symptoms = []
        self.qa_log = []                      # Now stores full descriptive Q&A history
        self.score_history = []
        self.final_diagnosis = None
        self.termination_reason = ""
        self.current_turn = 0

    def to_dict(self): return self.__dict__

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls.__new__(cls)
        obj.__dict__.update(d)
        return obj

    def save(self):
        path = os.path.join(SESSION_DIR, f"session_{self.session_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @staticmethod
    def load(session_id: str):
        path = os.path.join(SESSION_DIR, f"session_{session_id}.json")
        if not os.path.exists(path): raise FileNotFoundError()
        with open(path) as f: return DiagnosticSession.from_dict(json.load(f))


def normalise_name(name: str) -> str: return re.sub(r"\s+", " ", name.strip().lower())

def get_symptoms_for_disease(disease: str, min_weight: int = 1) -> list:
    d_node = normalise_name(disease)
    if not G.has_node(d_node):
        candidates = [n for n in G.nodes if d_node in n and G.nodes[n].get("node_type") == "disease"]
        if not candidates: return []
        d_node = candidates[0]
    return [succ for succ in G.successors(d_node) if G[d_node][succ].get("weight", 1) >= min_weight]

def compute_scores(candidate_diseases, confirmed, denied, ml_preds, hp: dict):
    ml_prior = {normalise_name(p.get("disease", "")): float(p.get("confidence", 0.5)) for p in ml_preds}
    results = []
    for disease in candidate_diseases:
        d_node = normalise_name(disease)
        graph_syms = set(get_symptoms_for_disease(d_node))
        confirmed_hit = sum([G[d_node][normalise_name(s)].get("weight", 1) * hp['confirmed_weight_mult'] 
                             for s in confirmed if G.has_edge(d_node, normalise_name(s))])
        denied_hit = sum([G[d_node][normalise_name(s)].get("weight", 1) * hp['denied_weight_mult'] 
                          for s in denied if G.has_edge(d_node, normalise_name(s))])
        prior = ml_prior.get(d_node, 0.3)
        total_syms = max(len(graph_syms), 1)
        raw_score = (prior * 5) + confirmed_hit - denied_hit
        results.append({"disease": disease, "score": round(max(raw_score / total_syms, 0.0), 5), "prior": round(prior, 4)})
        
    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        top_score = results[0]["score"]
        results = [r for r in results if r["score"] >= top_score * hp['min_score_to_keep']]
    return results[:hp['top_k_candidates']]

def pick_discriminating_symptom(ranked, asked_symptoms, top_n=4):
    top_diseases = [r["disease"] for r in ranked[:top_n]]
    disease_sym_sets = {d: set(get_symptoms_for_disease(d)) for d in top_diseases}
    all_candidate_syms = set().union(*disease_sym_sets.values()) - asked_symptoms
    if not all_candidate_syms: return None
    
    best_sym, best_ig, N = None, -1, len(top_diseases)
    for sym in all_candidate_syms:
        present_in = sum(1 for d in top_diseases if sym in disease_sym_sets[d])
        ig = present_in * (N - present_in)
        if ig > best_ig:
            best_ig, best_sym = ig, sym
    return best_sym

# def generate_question(symptom: str, top_diseases: list) -> str:
#     prompt = f"Current top candidate diagnoses: {', '.join(top_diseases[:4])}. Ask the patient ONE clear yes/no question to find out whether they have the symptom: '{symptom}'. Return ONLY the question text."
#     resp = groq_client.chat.completions.create(model=SMALL_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.6, max_tokens=80)
#     return resp.choices[0].message.content.strip().strip('"')

# def parse_yes_no(answer: str):
#     low = answer.lower().strip()
#     if any(t in low for t in ["yes", "yeah", "yep", "true"]): return True
#     if any(t in low for t in ["no", "nope", "not", "false"]): return False
#     return None


async def fetch_user_scan_context(user_id: Optional[str]) -> str:
    """Fetches the user's latest ML CV pipeline outputs from MongoDB to ground the LLM."""
    if not user_id: return "No prior computer vision scan data available."
    
    cursor = scans_col.find({"user_id": user_id}).sort("timestamp", -1).limit(2)
    scans = await cursor.to_list(length=None)
    
    if not scans: return "No prior computer vision scan data available."
    
    context_parts = []
    for i, s in enumerate(scans):
        context_parts.append(
            f"Scan {i+1} ({s['timestamp']}): ML detected '{s.get('detected_condition', 'Unknown')}' "
            f"on {s.get('detected_body_part', 'Unknown')} (Stress: {s.get('stress_index', 0)}%, "
            f"Hydration: {s.get('hydration_level', 0)}%)."
        )
    return " | ".join(context_parts)

def generate_contextual_question(symptom: str, top_diseases: list, session: DiagnosticSession) -> str:
    """Generates a dynamic, conversational question using full conversation state and ML history."""
    history_text = "\n".join([f"Q: {log['question']}\nA: {log['answer']}" for log in session.qa_log[-3:]])
    
    prompt = (
        f"You are an empathetic, expert AI clinical assistant. \n"
        f"User's initial doubt: '{session.user_query}'\n"
        f"User's recent ML Scan Data: {session.scan_context}\n"
        f"Current working hypotheses: {', '.join(top_diseases[:3])}\n"
        f"Recent conversation history:\n{history_text}\n\n"
        f"TASK: To narrow down the diagnosis, we need to know if the user is experiencing: '{symptom}'. "
        f"Ask the user ONE highly descriptive, conversational question to figure this out. "
        f"Do not sound robotic. Do not just ask a yes/no question; describe what the symptom might feel or look like so they can accurately answer. "
        f"Output ONLY the question text."
    )
    
    resp = groq_client.chat.completions.create(
        model=LLM_MODEL, # Upgraded to the larger model for better conversational nuance
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return resp.choices[0].message.content.strip().strip('"')

def analyze_user_answer(question: str, answer: str, target_symptom: str) -> Optional[bool]:
    """Uses LLM to evaluate free-text user responses instead of hardcoded keyword matching."""
    prompt = (
        f"Question asked to patient: '{question}'\n"
        f"Patient's exact reply: '{answer}'\n\n"
        f"Based on this exchange, does the patient CONFIRM experiencing the symptom '{target_symptom}', "
        f"DENY experiencing it, or is the answer UNCLEAR/unrelated?\n"
        f"Reply with exactly one word: CONFIRM, DENY, or UNCLEAR."
    )
    
    resp = groq_client.chat.completions.create(
        model=SMALL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10
    )
    
    result = resp.choices[0].message.content.strip().upper()
    if "CONFIRM" in result: return True
    if "DENY" in result: return False
    return None


# @app.post("/api/diagnose/start", tags=["Diagnostic Loop"])
# async def start_diagnosis(req: InitialDiagnosticRequest):
#     print("Received initial diagnostic request with symptoms:", req.symptom_text)
#     print("Using got request:", req.dict())
#     session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#     raw_syms = [s.strip() for s in re.split(r"[,;]+", req.symptom_text) if len(s.strip()) > 2]
#     ml_predictions = [{"disease": d, "confidence": 0.5} for d in raw_syms] 
#     session = DiagnosticSession(session_id, ml_predictions, req.hyperparams.model_dump())
#     candidate_diseases = list({n for n in G.nodes if G.nodes[n].get("node_type") == "disease"})[:session.hyperparams['top_k_candidates']]
    
#     ranked = compute_scores(candidate_diseases, [], [], ml_predictions, session.hyperparams)
#     session.score_history.append(ranked)
    
#     symptom_to_ask = pick_discriminating_symptom(ranked, set())
#     if not symptom_to_ask: raise HTTPException(status_code=400, detail="Cannot generate discriminating question from input.")
        
#     question_text = generate_question(symptom_to_ask, [r["disease"] for r in ranked[:4]])
#     session.current_turn = 1
#     session.save()
#     return {"session_id": session_id, "turn": session.current_turn, "question": question_text, "target_symptom": symptom_to_ask, "status": "ongoing"}

# @app.post("/api/diagnose/answer", tags=["Diagnostic Loop"])
# async def process_answer(req: AnswerRequest):
#     try: session = DiagnosticSession.load(req.session_id)
#     except FileNotFoundError: raise HTTPException(status_code=404, detail="Session not found")

#     last_symptom = session.qa_log[-1]["symptom"] if session.qa_log else None
#     if last_symptom:
#         interpreted = parse_yes_no(req.answer)
#         if interpreted is True: session.confirmed_symptoms.append(last_symptom)
#         elif interpreted is False: session.denied_symptoms.append(last_symptom)
#         session.qa_log.append({"turn": session.current_turn, "symptom": last_symptom, "answer": req.answer, "interpreted_yes": interpreted})

#     candidate_diseases = list({n for n in G.nodes if G.nodes[n].get("node_type") == "disease"})
#     ranked = compute_scores(candidate_diseases, session.confirmed_symptoms, session.denied_symptoms, session.ml_predictions, session.hyperparams)
#     session.score_history.append(ranked)

#     hp = session.hyperparams
#     is_confident = len(ranked) >= 2 and (ranked[0]["score"] / max(ranked[1]["score"], 1e-9) >= hp['confidence_threshold'])
    
#     if is_confident or session.current_turn >= hp['max_questions']:
#         session.termination_reason = "Confidence Reached" if is_confident else "Max Questions Reached"
#         session.final_diagnosis = ranked[0]
#         session.save()
#         report_path = os.path.join(SESSION_DIR, f"report_{session.session_id}.txt")
#         with open(report_path, "w") as f: f.write(f"FINAL DIAGNOSIS: {session.final_diagnosis['disease']} \nSCORE: {session.final_diagnosis['score']}")
#         return {"status": "complete", "diagnosis": session.final_diagnosis, "ranked_differentials": ranked[:3], "report_path": report_path}

#     asked_syms = {log["symptom"] for log in session.qa_log}
#     symptom_to_ask = pick_discriminating_symptom(ranked, asked_syms, top_n=4)
    
#     if not symptom_to_ask:
#         session.final_diagnosis = ranked[0]
#         session.save()
#         return {"status": "complete", "diagnosis": session.final_diagnosis}

#     question_text = generate_question(symptom_to_ask, [r["disease"] for r in ranked[:4]])
#     session.current_turn += 1
#     session.save()
#     return {"status": "ongoing", "turn": session.current_turn, "question": question_text, "target_symptom": symptom_to_ask}


@app.post("/api/diagnose/start", tags=["Diagnostic Loop"])
async def start_diagnosis(req: InitialDiagnosticRequest):
    print(f"Received initial diagnostic request: {req.symptom_text}")
    
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Extract raw initial concepts to seed the graph
    raw_syms = [s.strip() for s in re.split(r"[,;]+", req.symptom_text) if len(s.strip()) > 2]
    ml_predictions = [{"disease": d, "confidence": 0.5} for d in raw_syms] 
    
    # 1. Fetch ML Scan Context
    scan_context = await fetch_user_scan_context(req.user_id)
    
    # 2. Initialize Stateful Session
    session = DiagnosticSession(
        session_id=session_id, 
        ml_predictions=ml_predictions, 
        hyperparams=req.hyperparams.model_dump(),
        user_query=req.symptom_text,
        scan_context=scan_context
    )
    
    candidate_diseases = list({n for n in G.nodes if G.nodes[n].get("node_type") == "disease"})[:session.hyperparams['top_k_candidates']]
    ranked = compute_scores(candidate_diseases, [], [], ml_predictions, session.hyperparams)
    session.score_history.append(ranked)
    
    symptom_to_ask = pick_discriminating_symptom(ranked, set())
    if not symptom_to_ask: 
        raise HTTPException(status_code=400, detail="Cannot generate discriminating symptom from input.")
        
    # 3. Generate AI Contextual Question
    question_text = generate_contextual_question(symptom_to_ask, [r["disease"] for r in ranked[:4]], session)
    
    session.current_turn = 1
    session.qa_log.append({"turn": 1, "symptom": symptom_to_ask, "question": question_text, "answer": None, "interpreted_yes": None})
    session.save()
    
    return {
        "session_id": session_id, 
        "turn": session.current_turn, 
        "question": question_text, 
        "target_symptom": symptom_to_ask, 
        "status": "ongoing"
    }

@app.post("/api/diagnose/answer", tags=["Diagnostic Loop"])
async def process_answer(req: AnswerRequest):
    try: 
        session = DiagnosticSession.load(req.session_id)
    except FileNotFoundError: 
        raise HTTPException(status_code=404, detail="Session not found")

    # 1. Analyze previous answer using LLM
    last_log = session.qa_log[-1] if session.qa_log else None
    if last_log and last_log["answer"] is None:
        interpreted = analyze_user_answer(last_log["question"], req.answer, last_log["symptom"])
        
        if interpreted is True: session.confirmed_symptoms.append(last_log["symptom"])
        elif interpreted is False: session.denied_symptoms.append(last_log["symptom"])
        
        # Update the log with the user's actual reply and the LLM's interpretation
        last_log["answer"] = req.answer
        last_log["interpreted_yes"] = interpreted

    # 2. Re-compute hypothesis tree
    candidate_diseases = list({n for n in G.nodes if G.nodes[n].get("node_type") == "disease"})
    ranked = compute_scores(candidate_diseases, session.confirmed_symptoms, session.denied_symptoms, session.ml_predictions, session.hyperparams)
    session.score_history.append(ranked)

    # 3. Check early-exit (Confidence reached or Max turns hit)
    hp = session.hyperparams
    is_confident = len(ranked) >= 2 and (ranked[0]["score"] / max(ranked[1]["score"], 1e-9) >= hp['confidence_threshold'])
    
    if is_confident or session.current_turn >= hp['max_questions']:
        session.termination_reason = "Confidence Reached" if is_confident else "Max Questions Reached"
        session.final_diagnosis = ranked[0]
        session.save()
        
        report_path = os.path.join(SESSION_DIR, f"report_{session.session_id}.txt")
        with open(report_path, "w") as f: 
            f.write(f"FINAL DIAGNOSIS: {session.final_diagnosis['disease']} \nSCORE: {session.final_diagnosis['score']}\n")
            f.write(f"INITIAL QUERY: {session.user_query}\n")
            f.write(f"ML SCANS: {session.scan_context}\n")
            
        return {
            "status": "complete", 
            "diagnosis": session.final_diagnosis, 
            "ranked_differentials": ranked[:3], 
            "report_path": report_path
        }

    # 4. Pick next target symptom
    asked_syms = {log["symptom"] for log in session.qa_log}
    symptom_to_ask = pick_discriminating_symptom(ranked, asked_syms, top_n=4)
    
    if not symptom_to_ask:
        session.final_diagnosis = ranked[0]
        session.save()
        return {"status": "complete", "diagnosis": session.final_diagnosis}

    # 5. Generate descriptive next question using full state
    question_text = generate_contextual_question(symptom_to_ask, [r["disease"] for r in ranked[:4]], session)
    
    session.current_turn += 1
    session.qa_log.append({
        "turn": session.current_turn, 
        "symptom": symptom_to_ask, 
        "question": question_text, 
        "answer": None, 
        "interpreted_yes": None
    })
    session.save()
    
    return {
        "status": "ongoing", 
        "turn": session.current_turn, 
        "question": question_text, 
        "target_symptom": symptom_to_ask
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)