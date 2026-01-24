# Implementing PPGEO Pretraining on the nUWAy

This repository contains code and scripts for creating a custom unlabelled dataset from YouTube videos and running the PPGEO pretraining pipeline on the nUWAy dataset. The dataset generation process is designed to match the ACO dataset structure used in prior work.

---

## Creating Custom Unlabelled Dataset

### Setup

Navigate to the dataset workspace:
```bash
cd ppgeo_dataset
```

Create the Conda environment with all required dependencies:
```bash
conda env create -f ppgeo_env.yaml
```

Activate the environment:
```bash
conda activate ppgeo
```

---

### Preparing Video URLs

Create a text file containing YouTube URLs, with **one URL per line**, and save it as:
```
youtube_video_urls.txt
```

Convert the URL list into a numbered format:
```bash
nl -n rz -w 5 youtube_video_urls.txt > numbered_urls.txt
```

---

### Downloading YouTube Videos

Download videos on Linux using the provided script (uses `yt-dlp`):
```bash
bash download_youtube.sh
```

#### Notes on Cookies
- Firefox may block downloads due to cookie restrictions.
- Install the following Firefox extension to export cookies:  
  https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/
- Log in to YouTube and export cookies to a file named `cookies.txt`.
- Place `cookies.txt` inside the `ppgeo_dataset/` workspace.
- Cookies expire frequently and must be **re-exported every time** the download script is run.

---

### Frame Extraction

Extract video frames at **1 Hz** (default):
```bash
chmod +x extract_frames.sh
./extract_frames.sh
```

---

### Output Dataset Structure

The generated dataset follows the ACO dataset format:
```
ACO_ready_dataset/
├── dir-0/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── dir-1/
│   ├── 0.jpg
│   └── ...
```

---

### Optional: Convert Dataset to Grayscale

If your downstream task requires grayscale input (e.g., shuttle bus experiments), convert the dataset using:
```bash
python3 convert_dataset.py
```

---

## Running PPGEO Pipeline

Once the dataset has been created, it can be used directly for PPGEO pretraining.  
Follow the training and evaluation scripts provided in this repository to run the full PPGEO pipeline on the generated dataset.

---

## Notes

- This repository focuses on dataset creation and preparation for PPGEO pretraining.
- Ensure dataset structure matches the ACO format before running training.
- All scripts are intended for Linux-based systems.

---

## License

This project is intended for research use only.
