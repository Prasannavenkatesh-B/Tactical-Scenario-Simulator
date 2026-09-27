"""Generate the self-contained Google Colab training notebook for 5,000 iterations."""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🛡️ DRDO Tactical Scenario Simulator (TSS) — 5,000-Iteration Cloud Training Pipeline\n",
                "\n",
                "This notebook executes the full **Stage A (Low-Level Policies) → Freeze → Stage B (High-Level Commander)** reinforcement learning pipeline for **5,000 iterations per stage** on an NVIDIA T4 GPU runtime.\n",
                "\n",
                "### 📌 What This Pipeline Executes:\n",
                "1. **Stage A (5,000 iters)**: Trains 6 multi-domain policies (Air Dogfight, Air Evasion, Ground Engage, Ground Defend, Sea Engage, Sea Defend) across Curriculum Levels 1–5 with League self-play.\n",
                "2. **Freeze Barrier**: Mathematically freezes all low-level policy weights.\n",
                "3. **Stage B (5,000 iters)**: Trains the High-Level Commander policy for cross-domain orchestration.\n",
                "4. **Verification**: Re-evaluates 16 military doctrines via `validate_realism.py` and non-determinism via `verify_non_determinism.py`.\n",
                "5. **Google Drive Sync**: Automatically persists checkpoints to Google Drive every 100 iterations so no progress is ever lost."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Step 1: Verify Hardware & GPU Acceleration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "print('=' * 60)\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'CUDA Available:  {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU Device:      {torch.cuda.get_device_name(0)}')\n",
                "    !nvidia-smi\n",
                "else:\n",
                "    print('[!] WARNING: T4 GPU runtime not detected. In Colab, go to Runtime -> Change runtime type -> T4 GPU.')\n",
                "print('=' * 60)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💾 Step 2: Mount Google Drive for Persistent Storage\n",
                "*All checkpoints and metrics will be saved directly into Google Drive so that even if the session resets, your weights are safe.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "import os\n",
                "\n",
                "drive.mount('/content/drive')\n",
                "DRIVE_CKPT_DIR = '/content/drive/MyDrive/TSS_Checkpoints_5000'\n",
                "os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)\n",
                "print(f'[+] Google Drive mounted successfully!')\n",
                "print(f'[+] Checkpoint archive path: {DRIVE_CKPT_DIR}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📦 Step 3: Load Codebase (Via Google Drive ZIP or GitHub)\n",
                "\n",
                "*(Supports either `TSS_Project.zip` placed in Google Drive / uploaded to Colab, OR GitHub clone)*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import zipfile\n",
                "\n",
                "REPO_DIR = '/content/Tactical-Scenario-Simulator'\n",
                "drive_zip = '/content/drive/MyDrive/TSS_Project.zip'\n",
                "local_zip = '/content/TSS_Project.zip'\n",
                "\n",
                "if os.path.exists(drive_zip):\n",
                "    print(f'[*] Found {drive_zip} in Google Drive. Extracting...')\n",
                "    os.makedirs(REPO_DIR, exist_ok=True)\n",
                "    with zipfile.ZipFile(drive_zip, 'r') as zf:\n",
                "        zf.extractall(REPO_DIR)\n",
                "    %cd {REPO_DIR}\n",
                "    print(f'[+] Successfully loaded TSS codebase from Google Drive!')\n",
                "elif os.path.exists(local_zip):\n",
                "    print(f'[*] Found {local_zip}. Extracting...')\n",
                "    os.makedirs(REPO_DIR, exist_ok=True)\n",
                "    with zipfile.ZipFile(local_zip, 'r') as zf:\n",
                "        zf.extractall(REPO_DIR)\n",
                "    %cd {REPO_DIR}\n",
                "    print(f'[+] Successfully loaded TSS codebase from local upload!')\n",
                "elif os.path.exists(REPO_DIR):\n",
                "    print('[*] Directory already exists. Entering directory...')\n",
                "    %cd {REPO_DIR}\n",
                "else:\n",
                "    print('[*] Attempting GitHub clone...')\n",
                "    !git clone https://github.com/Prasannavenkatesh-B/Tactical-Scenario-Simulator.git {REPO_DIR}\n",
                "    %cd {REPO_DIR}\n",
                "\n",
                "print(f'[+] Current Working Directory: {os.getcwd()}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔧 Step 4: Install Dependencies & Run Verification Tests"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q pyyaml scipy scikit-learn matplotlib pandas pytest pygame\n",
                "!python -m pytest tests/test_smoke.py tests/test_env.py -v\n",
                "print('[+] Environment verified and ready for 5000-iteration training!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚀 Step 5: Execute 5,000-Iteration Staged Training Pipeline\n",
                "\n",
                "> **Important**: This runs Stage A (5,000 iterations) followed by Stage B (5,000 iterations).\n",
                "> Estimated duration on Colab: **6 to 8 hours**.\n",
                "> Every 100 iterations, checkpoints are automatically generated in `checkpoints/`.\n",
                "> We also start a background daemon that mirrors newly saved checkpoints into Google Drive every 2 minutes!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import threading\n",
                "import time\n",
                "import shutil\n",
                "import os\n",
                "\n",
                "# Background auto-sync function to Google Drive\n",
                "def sync_to_drive():\n",
                "    while True:\n",
                "        time.sleep(120)  # Every 2 minutes\n",
                "        if os.path.exists('checkpoints'):\n",
                "            try:\n",
                "                for item in os.listdir('checkpoints'):\n",
                "                    src = os.path.join('checkpoints', item)\n",
                "                    dst = os.path.join(DRIVE_CKPT_DIR, item)\n",
                "                    if os.path.isdir(src):\n",
                "                        shutil.copytree(src, dst, dirs_exist_ok=True)\n",
                "                    else:\n",
                "                        shutil.copy2(src, dst)\n",
                "                if os.path.exists('logs/metrics.csv'):\n",
                "                    shutil.copy2('logs/metrics.csv', os.path.join(DRIVE_CKPT_DIR, 'metrics.csv'))\n",
                "            except Exception:\n",
                "                pass\n",
                "\n",
                "sync_thread = threading.Thread(target=sync_to_drive, daemon=True)\n",
                "sync_thread.start()\n",
                "print('[+] Google Drive auto-sync daemon active (backing up every 2 minutes).')\n",
                "\n",
                "# Execute full 5000 iteration training\n",
                "!python scripts/train_all.py --iterations 5000\n",
                "\n",
                "# Final sync\n",
                "!cp -r checkpoints/* {DRIVE_CKPT_DIR}/\n",
                "!cp -r logs/* {DRIVE_CKPT_DIR}/\n",
                "print('[+] Training finished! All final models backed up to Google Drive.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎖️ Step 6: Validate Realism & Tactical Doctrines Post-Training\n",
                "*This evaluates the freshly trained models against the 16 DRDO military doctrines.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python scripts/validate_realism.py --checkpoint-dir checkpoints/final --num-episodes 100 --scenario-level 5 --output-dir reports/realism_5000\n",
                "!cp -r reports/realism_5000 {DRIVE_CKPT_DIR}/\n",
                "print('[+] Realism validation complete! Results saved to Google Drive.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎲 Step 7: Verify Non-Determinism & Stochasticity"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python scripts/verify_non_determinism.py --checkpoint-dir checkpoints/final --num-runs 100 --output-dir reports/non_determinism_5000\n",
                "!cp -r reports/non_determinism_5000 {DRIVE_CKPT_DIR}/\n",
                "print('[+] Non-determinism verification complete! Results saved to Google Drive.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 Step 8: Plot Training Curves (Win Rate & Loss)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import os\n",
                "\n",
                "metrics_file = 'logs/metrics.csv'\n",
                "if os.path.exists(metrics_file):\n",
                "    df = pd.read_csv(metrics_file)\n",
                "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "    \n",
                "    # Win rate curve\n",
                "    if 'win_rate' in df.columns:\n",
                "        ax1.plot(df['iteration'], df['win_rate'], color='#0284c7', lw=2, label='Curriculum Win Rate')\n",
                "        ax1.axhline(0.60, color='red', linestyle='--', label='DRDO 60% Threshold')\n",
                "        ax1.set_title('Curriculum Win Rate Progression (5,000 Iterations)', fontsize=12, fontweight='bold')\n",
                "        ax1.set_xlabel('Training Iterations')\n",
                "        ax1.set_ylabel('Win Rate')\n",
                "        ax1.grid(True, alpha=0.3)\n",
                "        ax1.legend()\n",
                "    \n",
                "    # Loss curve\n",
                "    loss_cols = [c for c in df.columns if 'loss' in c.lower()]\n",
                "    for col in loss_cols[:3]:\n",
                "        ax2.plot(df['iteration'], df[col], lw=1.5, label=col)\n",
                "    ax2.set_title('PPO/HHAPPO Loss Convergence', fontsize=12, fontweight='bold')\n",
                "    ax2.set_xlabel('Training Iterations')\n",
                "    ax2.set_ylabel('Loss')\n",
                "    ax2.grid(True, alpha=0.3)\n",
                "    ax2.legend()\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plot_path = os.path.join(DRIVE_CKPT_DIR, 'training_curves_5000.png')\n",
                "    plt.savefig(plot_path, dpi=300)\n",
                "    plt.savefig('reports/training_curves_5000.png', dpi=300)\n",
                "    plt.show()\n",
                "    print(f'[+] High-res plot saved to: {plot_path}')\n",
                "else:\n",
                "    print('Metrics file not found.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📦 Step 9: Package Trained Checkpoints into a Downloadable Archive"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!zip -r {DRIVE_CKPT_DIR}/TSS_Trained_5000_Deliverable.zip checkpoints/final logs/ reports/\n",
                "print('\\n' + '='*60)\n",
                "print(f'CONGRATULATIONS! Complete 5,000-iteration package is ready at:')\n",
                "print(f'{DRIVE_CKPT_DIR}/TSS_Trained_5000_Deliverable.zip')\n",
                "print('Download this file from your Google Drive and copy checkpoints/final back into your local TSS project!')\n",
                "print('='*60)"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_path = Path("TSS_Colab_Training_5000.ipynb")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created {target_path.resolve()}")
