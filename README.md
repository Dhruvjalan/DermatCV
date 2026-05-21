# BioCV Wellness Analytics Backend — Setup & Run Guide

> ⚠️ **DEMO PROTOTYPE** — All wellness scores are simulated and are **NOT** medical diagnoses.

---

## 1. Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| Python      | 3.10+          |
| pip         | 23+            |
| MongoDB Atlas account | free tier works |

---

## 2. Project Structure

```
project/
├── app.py                  ← main FastAPI backend (debugged)
├── requirements.txt
├── .env.example            ← copy to .env and fill values
├── .gitignore
├── checkpoints/            ← place .pth model files here (optional)
│   ├── body_classifier.pth
│   ├── disease_classifier.pth
│   ├── patch_autoencoder.pth
│   └── class_mapping.json
└── storage_vault/          ← auto-created; stores uploaded images
```

---

## 3. Step-by-Step Setup

### Step 1 — Clone / place files

```bash
mkdir biocv-backend && cd biocv-backend
# Place app.py, requirements.txt, .env.example here
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv

# Activate:
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users**: edit `requirements.txt` — uncomment the CUDA torch lines and
> comment out the CPU ones before running pip install.

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in **at minimum**:

```
MONGODB_URI=mongodb+srv://YOUR_USERNAME:YOUR_PASSWORD@YOUR_CLUSTER.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=wellness_analytics
```

**How to get your MongoDB URI:**
1. Log in to [MongoDB Atlas](https://cloud.mongodb.com)
2. Click your cluster → **Connect**
3. Choose **Drivers** → Python → copy the connection string
4. Replace `<password>` with your actual Atlas password

> ⚠️ **Never commit `.env` to git.** It's already in `.gitignore`.

### Step 5 — (Optional) Add model checkpoints

If you have trained `.pth` files, place them in `checkpoints/`:

```
checkpoints/
├── body_classifier.pth
├── disease_classifier.pth
├── patch_autoencoder.pth
└── class_mapping.json     ← optional; overrides default class lists
```

If checkpoints are absent, models run with **random weights** (demo mode — outputs are still generated).

### Step 6 — Run the server

```bash
# Development (with auto-reload)
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# OR use the __main__ block directly
python app.py
```

Server starts at: **http://localhost:8000**

---

## 4. API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/api/health` | Server + DB health check |
| POST   | `/api/users`  | Register new user |
| POST   | `/api/login`  | Authenticate user |
| GET    | `/api/users/{user_id}` | Get user profile |
| GET    | `/api/users/by-email/{email}` | Get user by email |
| POST   | `/api/analyze/{user_id}` | **Upload image → full scan** |
| GET    | `/api/history/{user_id}` | Get user scan history |
| GET    | `/api/admin/records` | All scan records (admin) |
| GET    | `/api/admin/summary` | Aggregate stats (admin) |
| DELETE | `/api/users/{user_id}` | Delete user + scans (admin) |

**Interactive docs:** http://localhost:8000/docs

---

## 5. Quick Test (curl)

```bash
# 1. Create a user
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"full_name": "Test User", "email": "test@example.com", "password": "secret"}'

# 2. Upload an image for scanning (replace USER_ID with value from step 1)
curl -X POST http://localhost:8000/api/analyze/USER_ID_HERE \
  -F "file=@/path/to/your/face_image.jpg"

# 3. Fetch scan history
curl http://localhost:8000/api/history/USER_ID_HERE

# 4. Admin summary
curl http://localhost:8000/api/admin/summary
```

---

## 6. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: MONGODB_URI is not set` | Missing `.env` file | Run `cp .env.example .env` and fill in your URI |
| `ServerSelectionTimeoutError` | Wrong URI or IP not whitelisted | In Atlas → Network Access → Add your IP (or 0.0.0.0/0 for dev) |
| `Authentication failed` | Wrong password in URI | Re-check Atlas password; URL-encode special chars |
| `ModuleNotFoundError: mediapipe` | Not installed | `pip install -r requirements.txt` inside your venv |
| `CUDA out of memory` | GPU too small | Set `CUDA_VISIBLE_DEVICES=""` to force CPU |
| `422 Unprocessable Entity` on /analyze | Not a valid image | Send a proper JPEG/PNG via multipart form |
| Port 8000 already in use | Another process | `uvicorn app:app --port 8001` |

---

## 7. Bugs Fixed (vs. original file)

| Severity | Issue | Fix |
|----------|-------|-----|
| 🔴 CRITICAL | Hardcoded MongoDB credentials in source | Removed — `.env` only |
| 🔴 CRITICAL | Masked password would fail auth | `.env` with real credentials required |
| 🟠 BUG | `@app.on_event` deprecated | Replaced with `lifespan` context manager |
| 🟠 BUG | GradCAM hooks never fire | `requires_grad_(True)` on input tensor |
| 🟠 BUG | FC layer dim mismatch in DiseaseClassifier | Corrected to `64 * 28 * 28` |
| 🟠 BUG | Degenerate patch crashes AE | Guard + fallback to full image |
| 🟠 BUG | Unsafe latent vector indexing | `.flatten()` before index access |
| 🟠 BUG | `uvicorn` target `"app.py:app"` wrong | Corrected to `"app:app"` |
| 🟡 WARN | Passwords stored in plaintext | Hashed with `passlib[bcrypt]` |
| 🟡 WARN | CORS hardcoded to localhost:3000 | Now reads from `CORS_ORIGINS` env var |
| 🟡 WARN | Unused imports (`Depends`, `ObjectId`) | Removed |
