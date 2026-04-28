# AgriAI Disease Detection and Advisory App

This repository contains a Streamlit-based agriculture assistant with plant disease detection, crop recommendation, fertilizer advice, and market forecasting.

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

3. Start the Streamlit app:

```powershell
streamlit run app.py
```

## Deployment

- Use `requirements.txt` for package installation.
- On Streamlit Cloud, set the entrypoint to `app.py`.
- Keep the `models/disease_detection` folder in the repository if the disease scan feature requires the saved model files.

## Notes

- The app is built for `Streamlit` and `TensorFlow`.
- If deployment fails due to TensorFlow compatibility, try pinning a CPU build like `tensorflow-cpu==2.12.0` or use Python 3.11.
- `.gitignore` excludes local virtual environments, dataset folders, and temporary files.

cd /d d:\Capstone
git remote set-url origin https://github.com/Pavand0515/Crop-Reccomendation-and-Crop-Health-Mointoring.git
git push -u origin master:maincd /d d:\Capstone
git remote set-url origin https://github.com/Pavand0515/Crop-Reccomendation-and-Crop-Health-Mointoring.git
git push -u origin master:main