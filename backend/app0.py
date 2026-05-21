import os
import json
import cv2
import uuid
import sqlite3
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="BioCV Enterprise Wellness Analytics Core",
    version="2.0.0",
    description="Backend pipeline with integrated computer vision, SQLite tracking, and Admin panels."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
STORAGE_DIR = "storage_vault"
DB_PATH = "wellness_matrix.db"

# Ensure runtime directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Serve processing storage to allow frontend dashboard previews
app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # 1. Users Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        # 2. Scans History Table (Stores CV metrics, Image paths, and RAG logs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scans (
                scan_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                original_image_path TEXT NOT NULL,
                processed_image_path TEXT NOT NULL,
                detected_body_part TEXT,
                blur_score INTEGER,
                brightness_score INTEGER,
                stress_index INTEGER,
                fatigue_index INTEGER,
                hydration_level INTEGER,
                overall_wellness_score INTEGER,
                recommendations TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        conn.commit()
    print("🗄️ Database Hub: Relational tables initialized successfully.")

# Pydantic validation rules for creating new profiles
class UserCreate(BaseModel):
    full_name: str
    email: str

# =====================================================================
# NEURAL NETWORK LAYER ARCHITECTURES
# =====================================================================
class BodyPartClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

class DiseaseClassifierWithGradCAM(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 28 * 28, num_classes)
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad): self.gradients = grad

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        self.activations = x
        if x.requires_grad: x.register_hook(self.activations_hook)
        return self.fc(self.pool(x).view(x.size(0), -1))

class NormalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 14 * 14, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32 * 14 * 14), nn.ReLU(), nn.Unflatten(1, (32, 14, 14)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        latent = F.normalize(self.encoder(x), p=2, dim=1)
        return latent, self.decoder(latent)

# Global variables mapping structural weights across active memory
BODY_CLASSES = ["Face", "Skin Hand", "Eye", "Forehead", "Cheek", "Neck", "Arm", "Leg"]
DISEASE_CLASSES = ["Healthy Base", "Surface Fatigue", "Dehydration Zone", "Vascular Flush"]
body_model, disease_model, ae_model = None, None, None

@app.on_event("startup")
def startup_pipeline_activation():
    global body_model, disease_model, ae_model, BODY_CLASSES, DISEASE_CLASSES
    initialize_database()
    
    # Initialize basic instances
    body_model = BodyPartClassifier(num_classes=len(BODY_CLASSES)).to(DEVICE)
    disease_model = DiseaseClassifierWithGradCAM(num_classes=len(DISEASE_CLASSES)).to(DEVICE)
    ae_model = NormalAutoencoder().to(DEVICE)

    # Graceful fallback configuration checks if checkpoints aren't created yet
    try:
        mapping_path = os.path.join(CHECKPOINT_DIR, "class_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            BODY_CLASSES = mapping["body_classes"]
            DISEASE_CLASSES = mapping["disease_classes"]
            
        body_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "body_classifier.pth"), map_location=DEVICE))
        disease_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "disease_classifier.pth"), map_location=DEVICE))
        ae_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "patch_autoencoder.pth"), map_location=DEVICE))
        print("🛰️ Deep Learning Core: Fully loaded state dictionaries.")
    except Exception as e:
        print(f"⚠️ Warning: Checkpoint keys missing or unreadable ({e}). Initializing models with randomized parameters for testing.")
        
    body_model.eval()
    disease_model.eval()
    ae_model.eval()

def run_quality_checks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = int(cv2.Laplacian(gray, cv2.CV_64F).var())
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    brightness_score = int(np.mean(cv2.split(lab)[0]))
    return blur_score, brightness_score

def compute_gradcam_patches(image_tensor, model, N=4, P=4):
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    patch_h, patch_w = H // N, W // N
    
    outputs = model(image_tensor)
    target_class = torch.argmax(outputs, dim=1).item()
    score = outputs[0, target_class]
    
    model.zero_grad()
    score.backward(retain_graph=True)
    
    combined_heatmap = torch.zeros((H, W)).to(DEVICE)
    if model.gradients is not None and model.activations is not None:
        pooled_grads = torch.mean(model.gradients, dim=[2, 3], keepdim=True)
        gradcam = F.relu(torch.sum(pooled_grads * model.activations, dim=1).squeeze(0))
        combined_heatmap = F.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear').squeeze().detach()

    if combined_heatmap.max() > 0:
        combined_heatmap /= combined_heatmap.max()

    heatmap_cpu = combined_heatmap.cpu().numpy()
    patch_metrics = []
    for i in range(N):
        for j in range(N):
            y1, y2 = i * patch_h, (i + 1) * patch_h
            x1, x2 = j * patch_w, (j + 1) * patch_w
            patch_metrics.append((heatmap_cpu[y1:y2, x1:x2].mean().item(), (x1, y1, x2, y2)))
            
    patch_metrics.sort(key=lambda x: x[0], reverse=True)
    return patch_metrics[:P]


@app.post("/api/users", tags=["User Engine"])
def create_user_profile(user: UserCreate):
    """Generates an unique user state context for dashboard assignment."""
    user_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (user_id, user.full_name, user.email, created_at))
            conn.commit()
        return {"status": "success", "user_id": user_id, "full_name": user.full_name, "email": user.email}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Profile containing this email registration already exists.")

@app.post("/api/analyze/{user_id}", tags=["Core Pipeline"])
async def process_and_log_scan(user_id: str, file: UploadFile = File(...)):
    """
    Accepts raw multi-part forms, isolates facial regions, computes computer vision pipelines,
    extracts dummy wellness vectors, and logs entries into the relational database architecture.
    """
    # 1. Verify user profile exists
    with get_db_connection() as conn:
        user_record = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user_record:
        raise HTTPException(status_code=404, detail="User account signature not found in system storage database.")

    # 2. File verification checks
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    native_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if native_img is None:
        raise HTTPException(status_code=400, detail="Invalid graphic matrix stream sent.")

    scan_id = f"SCAN_{str(uuid.uuid4())[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save original image asset to server vault
    orig_filename = f"{scan_id}_orig.jpg"
    orig_filepath = os.path.join(STORAGE_DIR, orig_filename)
    cv2.imwrite(orig_filepath, native_img)

    # 3. Quality Analysis Logic Check
    blur_score, brightness_score = run_quality_checks(native_img)
    h_nat, w_nat, _ = native_img.shape

    # 4. MediaPipe Region-of-Interest (ROI) Extraction
    mp_face = mp.solutions.face_detection
    rgb_img = cv2.cvtColor(native_img, cv2.COLOR_BGR2RGB)
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as face_det:
        results = face_det.process(rgb_img)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            xmin = max(0, int(bbox.xmin * w_nat))
            ymin = max(0, int(bbox.ymin * h_nat))
            xmax = min(w_nat, xmin + int(bbox.width * w_nat))
            ymax = min(h_nat, ymin + int(bbox.height * h_nat))
            skin_roi = native_img[ymin:ymax, xmin:xmax]
            regions_log = {"face_detected": True, "bounding_box": [xmin, ymin, xmax, ymax]}
        else:
            skin_roi = native_img.copy()
            regions_log = {"face_detected": False, "bounding_box": [0, 0, w_nat, h_nat]}

    skin_roi_resized = cv2.resize(skin_roi, (224, 224))
    full_resized = cv2.resize(native_img, (224, 224))

    # Convert tensors for network inference
    input_tensor = torch.tensor(skin_roi_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    input_tensor.requires_grad = True
    body_tensor = torch.tensor(full_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

    # 5. Core Model Executions
    with torch.no_grad():
        body_out = body_model(body_tensor)
        detected_body_part = BODY_CLASSES[torch.argmax(body_out).item()]

    top_patches = compute_gradcam_patches(input_tensor, disease_model, N=4, P=4)

    # Paint annotations onto structural preview outputs
    annotated_img = skin_roi_resized.copy()
    for rank, (intensity, coords) in enumerate(top_patches, 1):
        x1, y1, x2, y2 = coords
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, f"P{rank}", (x1+4, y1+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    proc_filename = f"{scan_id}_proc.jpg"
    proc_filepath = os.path.join(STORAGE_DIR, proc_filename)
    cv2.imwrite(proc_filepath, annotated_img)

    # 6. Extract Autoencoder Latent Spaces for Simulated Metrics Calculation
    x1, y1, x2, y2 = top_patches[0][1]
    target_patch = input_tensor[:, :, y1:y2, x1:x2]
    ae_in = F.interpolate(target_patch, size=(56, 56))
    with torch.no_grad():
        latent_vec, _ = ae_model(ae_in)
    
    # Deterministic generation using vector space features
    v_space = latent_vec.squeeze().cpu().numpy()
    stress_index = int(abs(v_space[0] * 100) % 100)
    fatigue_index = int(abs(v_space[1] * 100) % 100)
    hydration_level = int(100 - (abs(v_space[2] * 100) % 100))
    overall_wellness_score = int((hydration_level + (100 - stress_index) + (100 - fatigue_index)) // 3)

    # Lifestyle recommendations mapping engine
    recommendations_pool = [
        "Incorporate 4-7-8 deep breathing exercises to downregulate cortisol responses.",
        "Optimize sleep patterns and implement cooling eye blocks to manage under-eye vascular pooling.",
        "Target a minimum baseline of 3 liters of structured water hydration fluid lines daily."
    ]

    # Save to SQLite Database Instance
    with get_db_connection() as conn:
        conn.execute('''
            INSERT INTO scans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scan_id, user_id, f"/static/{orig_filename}", f"/static/{proc_filename}",
            detected_body_part, blur_score, brightness_score, stress_index, fatigue_index,
            hydration_level, overall_wellness_score, json.dumps(recommendations_pool), timestamp
        ))
        conn.commit()

    # 7. Package structured layout for dashboard ingestion (Requirement 4)
    return {
        "status": "success",
        "scan_id": scan_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "disclaimer": "⚠️ DEMO PROTOTYPE OUTPUT: This system provides proxy wellness metrics. It is not an entry for medical diagnoses or treatments.",
        "dashboard_data": {
            "image_previews": {
                "original_url": f"/static/{orig_filename}",
                "processed_url": f"/static/{proc_filename}"
            },
            "segmentation": {
                "detected_body_region": detected_body_part,
                "structural_metadata": regions_log
            },
            "quality_metrics": {
                "sharpness_index": blur_score,
                "brightness_index": brightness_score,
                "status": "Acceptable Processing Threshold" if blur_score > 50 else "Suboptimal Focus"
            },
            "wellness_biometrics": {
                "stress_index": stress_index,
                "fatigue_index": fatigue_index,
                "hydration_level": hydration_level,
                "composite_wellness_score": overall_wellness_score
            },
            "actionable_interventions": recommendations_pool
        }
    }

@app.get("/api/history/{user_id}", tags=["Core Pipeline"])
def get_user_scan_history(user_id: str):
    """Fetches full historical timeline profiles for a given User UUID."""
    with get_db_connection() as conn:
        records = conn.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY timestamp DESC", (user_id,)).fetchall()
    
    history_list = []
    for r in records:
        history_list.append({
            "scan_id": r["scan_id"],
            "timestamp": r["timestamp"],
            "body_part": r["detected_body_part"],
            "wellness_score": r["overall_wellness_score"],
            "biometrics": {"stress": r["stress_index"], "fatigue": r["fatigue_index"], "hydration": r["hydration_level"]},
            "quality": {"blur": r["blur_score"], "brightness": r["brightness_score"]},
            "urls": {"original": r["original_image_path"], "processed": r["processed_image_path"]},
            "recommendations": json.loads(r["recommendations"])
        })
    return {"user_id": user_id, "total_scans": len(history_list), "history": history_list}


@app.get("/api/admin/records", tags=["Admin Portal CONTROL"])
def fetch_system_wide_audit_log():
    """Admin endpoint to monitor uploaded records across all profiles."""
    with get_db_connection() as conn:
        scans_query = conn.execute('''
            SELECT s.*, u.full_name, u.email 
            FROM scans s 
            JOIN users u ON s.user_id = u.user_id 
            ORDER BY s.timestamp DESC
        ''').fetchall()
        
    audit_logs = []
    for row in scans_query:
        audit_logs.append({
            "scan_id": row["scan_id"],
            "timestamp": row["timestamp"],
            "user_details": {"id": row["user_id"], "name": row["full_name"], "email": row["email"]},
            "vision_output": {"body_part": row["detected_body_part"], "blur": row["blur_score"], "brightness": row["brightness_score"]},
            "biometric_scores": {"stress": row["stress_index"], "fatigue": row["fatigue_index"], "hydration": row["hydration_level"], "composite": row["overall_wellness_score"]},
            "artifacts": {"orig_url": row["original_image_path"], "proc_url": row["processed_image_path"]}
        })
    return {"admin_audit_log_records": audit_logs}

@app.get("/api/admin/summary", tags=["Admin Portal CONTROL"])
def fetch_system_analytics_summary():
    """Compiles dashboard system usage metrics for the Admin summary overview cards."""
    with get_db_connection() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        avg_wellness = conn.execute("SELECT AVG(overall_wellness_score) FROM scans").fetchone()[0] or 0
        
    return {
        "system_status": "ONLINE",
        "total_registered_users": total_users,
        "total_scans_processed": total_scans,
        "average_population_wellness_score": round(avg_wellness, 1)
    }


@app.get("/api/users/by-email/{email}")
def get_user_by_email(email: str):
    with get_db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user["user_id"], "full_name": user["full_name"], "email": user["email"]}

@app.get("/api/users/{user_id}")
def get_user_by_id(user_id: str):
    with get_db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user["user_id"], "full_name": user["full_name"], "email": user["email"]}


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)