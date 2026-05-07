# AgriAI Disease Detection and Advisory App

This repository contains a React + FastAPI agriculture assistant with plant disease detection, crop recommendation, fertilizer advice, and market forecasting.

## Run locally

1. Create and activate a Python environment (recommended Python 3.12):

```powershell
python -m venv .venv_tf
.\.venv_tf\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the FastAPI backend:

```powershell
uvicorn app:app --reload
```

4. Open the built frontend through FastAPI:

```text
http://127.0.0.1:8000/
```

For frontend development, run this in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

## Deployment

- Use `requirements.txt` for package installation.
- Build the React frontend with `npm run build` inside `frontend/`.
- Start the API with `uvicorn app:app --host 0.0.0.0 --port 8000`.
- Keep the `models/disease_detection` folder in the repository if the disease scan feature requires the saved model files.

## Notes

- The backend is built with `FastAPI` and `TensorFlow`.
- The frontend lives in `frontend/src`.
- If deployment fails due to TensorFlow compatibility, try pinning a CPU build like `tensorflow-cpu==2.12.0` or use Python 3.11.
- `.gitignore` excludes local virtual environments, dataset folders, and temporary files.

