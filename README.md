# Plant Disease Classification with a Convolutional Neural Network

This repository contains a Jupyter Notebook (PlantClassification.ipynb) that trains and evaluates a convolutional neural network (CNN) for plant disease classification. The notebook walks through data download (via the Kaggle API), preprocessing, model building, training, evaluation, and visualization of results so you can reproduce the experiments.

Files in this repository
- PlantClassification.ipynb — the main notebook with the full pipeline (data -> model -> evaluation).
- kaggle.json — a Kaggle API token file (replace with your own, see below).

Important note about kaggle.json
- A kaggle.json file is present in the repository listing. This file contains Kaggle API credentials — for security you should replace it with your own token or remove it before publishing public repos. The instructions below show how to use your Kaggle token safely.

Prerequisites
- Python 3.8+ (recommended)
- Recommended: a CUDA-enabled GPU with recent NVIDIA drivers for training. CPU training will work but is much slower.
- Packages used (the notebook cells also show the exact imports):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow` (or `tensorflow-gpu`) / `keras`
  - `opencv-python` (cv2) or `PIL` (Pillow)
  - `kaggle`
  - `tqdm`

Quick setup (pip)
1. Create and activate a virtual environment:
   - `python -m venv venv`
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`

2. Install packages:
   - `pip install --upgrade pip`
   - `pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python pillow kaggle tqdm jupyter`

(Optionally create a requirements.txt from the above list to pin versions.)

## Kaggle dataset download

The notebook uses the Kaggle API to download the dataset. Steps to set up Kaggle access:

1. Get your Kaggle API token:
   - Go to [Kaggle](https://www.kaggle.com/), sign in, then Account -> Create New API Token. This downloads a kaggle.json file.

2. Place your kaggle.json in a secure location:
   - Recommended: move the file to `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json (Windows)`.
   - Or set the environment variable `KAGGLE_CONFIG_DIR` to the directory containing kaggle.json.

3. Example command to download a dataset (replace <owner>/<dataset> with the dataset slug used in the notebook):
   - `kaggle datasets download -d <owner>/<dataset> -p data/ --unzip`

If you prefer, you can run the dataset download cells inside PlantClassification.ipynb (these cells use the Kaggle API). If the repository's kaggle.json is present, replace it with your own token before use.

How to reproduce results (recommended)
1. Open the notebook
   - Run Jupyter locally:
     - jupyter notebook PlantClassification.ipynb
   - Or open in Google Colab: upload your `kaggle.json` (or set Kaggle token in Colab) and run the cells.

2. Configure runtime
   - If training, select a GPU runtime (Colab) or ensure TensorFlow detects your GPU locally.
   - If you want to reproduce exact runs, set random seeds in the notebook (cells near the top) for Python, NumPy, TensorFlow, and any other used libraries.

3. Run the notebook cells in order
   - Data download / preparation cells (creates a data/ or dataset/ folder).
   - Preprocessing and augmentation cells (image resizing, normalization, train/val split).
   - Model building cell (CNN architecture).
   - Training cell(s) — will save model weights and training history.
   - Evaluation and visualization cells (confusion matrix, sample predictions, metrics).

4. Execute the entire notebook programmatically (optional)
   - jupyter nbconvert --to notebook --execute PlantClassification.ipynb --ExecutePreprocessor.timeout=0 --output executed.ipynb
   - Increase timeout if training takes long, or run interactive in Jupyter/Colab.

What the notebook produces (expected outputs)
- Trained model weights (e.g., models/model.h5 or similar — check the notebook’s save paths).
- Training history / logs (loss, accuracy plots).
- Evaluation metrics and confusion matrix plots.
- Example predictions on validation/test images.

Reproducibility tips
- Use the same package versions (TensorFlow, Keras) if you need bit-for-bit reproducibility.
- Fix seeds (numpy.random.seed, random.seed, tensorflow.random.set_seed).
- Ensure consistent image preprocessing (resize, normalization).
- Record hardware used (GPU model) and batch size / learning rate.

Common issues and troubleshooting
- Kaggle credentials errors:
  - Ensure kaggle.json is valid and located at ~/.kaggle/kaggle.json or KAGGLE_CONFIG_DIR is set.
  - File permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac).
- Out-of-memory (OOM) on GPU:
  - Reduce batch size, reduce image resolution, or use a smaller model.
- Slow training on CPU:
  - Use a GPU runtime (Colab or local) for reasonable training times.

Notes about dataset and licensing
- The dataset used in the notebook is downloaded from Kaggle via the Kaggle API. Ensure you follow the dataset license and attribution requirements listed on the dataset page.
- Replace or remove any sensitive credential files (like kaggle.json) before making your repo public.

Suggested next steps
- Hyperparameter tuning: try different optimizers, learning rates, batch sizes, and augmentations.
- Transfer learning: experiment with pretrained backbones (MobileNet, EfficientNet, ResNet).
- Cross-validation: implement k-fold cross-validation for more robust evaluation.
- Export model for inference: save model in SavedModel format and create a small inference script.

Contact / Questions
If you have questions about reproducing the notebook or want help adapting it to a different dataset or environment, open an issue or contact the repository owner.
