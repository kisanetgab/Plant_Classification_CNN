# Plant Disease Classification CNN

A convolutional neural network for detecting plant diseases from leaf images, trained on the PlantVillage dataset. This repo contains the full pipeline — data download, preprocessing, model building, training, and evaluation — in `PlantClassification.ipynb`, along with the write-up below.

**Results at a glance:** a baseline CNN reached ~86% validation accuracy; an improved architecture with batch normalization, dropout, and data augmentation pushed that to 90%+. See [Results](#results) for details.

## Table of Contents

- [Introduction](#introduction)
- [Setup & Reproduction](#setup--reproduction)
- [Methodology & Data](#methodology--data)
  - [Baseline CNN](#baseline-cnn)
  - [Improved CNN](#improved-cnn)
  - [Training](#training)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [Suggested Next Steps](#suggested-next-steps)

## Introduction

Plant diseases pose a significant threat to global food production, reducing crop yield and lowering the economic stability of farming communities worldwide. Early detection is crucial, yet many farmers lack immediate access to expert diagnosis. As agriculture increasingly intersects with technology, machine learning offers a solution that involves scalable, accurate, and accessible plant disease identification.

This project focuses on developing a plant disease classification system that uses a convolutional neural network to detect diseases from leaf images. By training a neural model on thousands of labeled plant-leaf images from the PlantVillage dataset, the goal is to build an automated tool capable of recognizing visual disease patterns in real time. The project explores the capabilities of modern deep learning in agricultural contexts and contributes to a broader vision of using AI to support food security and empower growers with accessible diagnostic tools.

> **Central question:** Can a custom-built deep learning model accurately classify diseases from raw image inputs, and what architectural or preprocessing choices most strongly influence that accuracy?

## Setup & Reproduction

### Files in this repository

- `PlantClassification.ipynb` — the main notebook with the full pipeline (data → model → evaluation)

### Prerequisites

- Python 3.8+
- Recommended: a CUDA-enabled GPU with recent NVIDIA drivers for training (CPU training works but is much slower)
- Packages used (also shown in the notebook's import cells): `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` (or `tensorflow-gpu`) / `keras`, `opencv-python` (`cv2`) or `Pillow`, `kaggle`, `tqdm`

### Quick setup (pip)

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python pillow kaggle tqdm jupyter
```

(Optionally freeze the above into a `requirements.txt` to pin versions.)

### Kaggle dataset download

The notebook uses the Kaggle API to download the PlantVillage dataset. To set up access:

1. **Get your Kaggle API token** — sign in to Kaggle, then go to **Account → Create New API Token**. This downloads a `kaggle.json` file.
2. **Place your `kaggle.json` in a secure location:**
   - Recommended: `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
   - Or set the `KAGGLE_CONFIG_DIR` environment variable to the directory containing it
   - On Linux/Mac, lock down permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. **Download the dataset:**
   ```bash
   kaggle datasets download -d <owner>/<dataset> -p data/ --unzip
   ```
   (replace `<owner>/<dataset>` with the dataset slug used in the notebook)

   Alternatively, run the dataset-download cells inside `PlantClassification.ipynb` directly — they call the Kaggle API the same way.

   > **Note:** No credentials are stored in this repository. You'll need to generate and supply your own `kaggle.json` token to run the notebook.

### How to reproduce results

1. **Open the notebook**
   ```bash
   jupyter notebook PlantClassification.ipynb
   ```
   or open it in Google Colab and set your Kaggle token there.
2. **Configure runtime** — select a GPU runtime (Colab) or confirm TensorFlow detects your local GPU. For exact reproduction, set random seeds near the top of the notebook (Python, NumPy, TensorFlow).
3. **Run the cells in order:**
   - Data download / preparation (creates a `data/` or `dataset/` folder)
   - Preprocessing and augmentation (resizing, normalization, train/val split)
   - Model building (CNN architecture)
   - Training (saves model weights and training history)
   - Evaluation and visualization (confusion matrix, sample predictions, metrics)
4. **Optional — run the whole notebook headlessly:**
   ```bash
   jupyter nbconvert --to notebook --execute PlantClassification.ipynb --ExecutePreprocessor.timeout=0 --output executed.ipynb
   ```
   Increase the timeout if training takes a while.

### What the notebook produces

- Trained model weights (check the notebook's save paths)
- Training history / logs (loss and accuracy plots)
- Evaluation metrics and a confusion matrix
- Example predictions on validation/test images

### Reproducibility tips

- Match package versions (TensorFlow, Keras) for bit-for-bit reproducibility
- Fix seeds (`numpy.random.seed`, `random.seed`, `tensorflow.random.set_seed`)
- Keep image preprocessing consistent (resize, normalization)
- Record hardware used (GPU model) and batch size / learning rate

### Troubleshooting

- **Kaggle credential errors** — confirm `kaggle.json` is valid and located at `~/.kaggle/kaggle.json`, or that `KAGGLE_CONFIG_DIR` is set correctly.
- **Out-of-memory (OOM) on GPU** — reduce batch size, lower image resolution, or use a smaller model.
- **Slow training on CPU** — switch to a GPU runtime (Colab or local) for reasonable training times.

### Dataset licensing

The dataset is downloaded from Kaggle via the Kaggle API. Follow the license and attribution requirements listed on the dataset's Kaggle page.

## Methodology & Data

This project uses the **PlantVillage dataset**, widely used in plant-disease classification research. It contains over 50,000 labeled images across numerous plant species and disease categories. The dataset was downloaded through Kaggle, and images were organized into class-specific subdirectories upon extraction for direct loading with TensorFlow's `image_dataset_from_directory` utility.

Before being passed into the neural network, images were standardized for consistency and efficiency:

| Preprocessing step | Value |
|---|---|
| Resizing | 128 x 128 pixels |
| Batch size | 64 |
| Pixel rescaling | [0, 0.5] range |

Some models also incorporated **data augmentation** — random flips, rotations, and zooms — to reduce overfitting and increase robustness.

### Baseline CNN

A simple, sequential neural network that allowed for quick training and provided a foundation for understanding how more complex architectures might perform.

```
Conv2D (32 filters)
  → MaxPooling
  → Flatten
  → Dense (128, ReLU)
  → Dense (softmax output)
```

### Improved CNN

After evaluating the baseline network, an enhanced model was developed with deeper feature extraction capabilities and regularization techniques:

- Three convolutional blocks, each with Conv2D layers, batch normalization, ReLU activation, and max pooling
- Global average pooling for dimensionality reduction
- Dense layers with dropout for regularization
- A data augmentation pipeline for better generalization

This architecture allows the model to learn more detailed spatial patterns associated with specific plant diseases.

### Training

| Component | Choice |
|---|---|
| Optimizer | Adam |
| Loss | Sparse categorical crossentropy |
| Regularization | Early stopping, LR reduction on plateau |

Model performance was tracked across epochs using validation accuracy and loss, visualized with custom plotting functions. Post-training, predictions on the validation set were used to compute overall accuracy, weighted precision, recall, F1-score, and a confusion matrix heatmap to diagnose class-specific performance issues.

## Results

After training both the baseline and improved CNN models on the PlantVillage dataset, the project produced clear evidence of the value of deeper architectures, data augmentation, and regularization in improving classification performance.

| Model | Validation Accuracy | Notes |
|---|---|---|
| Baseline CNN | ~86% | Reliable but limited generalization; captures fundamental leaf texture/color patterns |
| Improved CNN | 90%+ | Outperforms baseline in nearly all metrics, especially on visually similar disease classes |

## Discussion

The results highlight a clear pattern: model complexity and input variability significantly influence classification accuracy in plant disease detection. The baseline CNN successfully captured fundamental aspects of leaf texture and coloration, allowing it to perform reasonably well. However, its limited depth and lack of regularization resulted in faster overfitting, particularly in disease categories with overlapping visual characteristics — a common issue in image classification systems where surface-level features are insufficient to distinguish subtle disease markers.

The improved CNN addressed these limitations by incorporating deeper convolutional blocks and batch normalization, which stabilized learning, reduced internal covariate shift, and allowed the model to generalize more effectively. Dropout also lowered overfitting, especially in the dense layers where the baseline model showed the greatest tendency to memorize training images.

The most impactful addition to the network was likely data augmentation. By introducing random flips, rotations, and zoom operations, the model learned to recognize diseases across a wider range of orientations and visual noise conditions — improving validation accuracy and better simulating real-world image variability, which the original dataset lacks since many PlantVillage images are captured in ideal conditions. Hyperparameter tuning — experimenting with learning rates, dropout rates, and filter counts — helped strike a balance between model complexity and generalization ability.

**Challenges:**

- **Dataset imbalance** — some disease classes contained significantly fewer examples. Batch normalization and dropout in the improved model helped mitigate this, but class imbalance still contributed to some confusion matrix patterns.
- **Computational limitations** on Google Colab required more careful memory management and more efficient dataset pipelines (`cache()`, `prefetch()`, and batching techniques), which helped reduce training time and stabilize model performance.

Overall, the models align with widely reported results showing that CNN-based approaches can reach high accuracy on the PlantVillage dataset. As with many studies, results also point to the limitations of training exclusively on highly controlled images and illustrate why real-world deployment requires additional domain-specific data.

## Conclusion

This project successfully developed and evaluated convolutional neural networks for plant disease classification using the PlantVillage dataset. Through systematic experimentation, the findings demonstrate that a deeper and more carefully regularized CNN architecture outperforms a simpler baseline model. The improved CNN achieved high validation accuracy, strong precision/recall performance, and enhanced generalization due to data augmentation, batch normalization, and dropout — highlighting the importance of architectural sophistication and training strategy when dealing with diverse, real-world visual classification tasks.

Deep learning provides a powerful foundation for developing accessible diagnostic tools in agriculture. With further work — such as incorporating field images, addressing dataset imbalance, and possibly exploring transfer learning from larger pretrained models — this approach has strong potential to support scalable, real-world plant disease detection systems that empower farmers, improve crop yields, and contribute to global food security.

## Suggested Next Steps

- **Hyperparameter tuning** — try different optimizers, learning rates, batch sizes, and augmentations
- **Transfer learning** — experiment with pretrained backbones (MobileNet, EfficientNet, ResNet)
- **Cross-validation** — implement k-fold cross-validation for more robust evaluation
- **Export for inference** — save the model in SavedModel format and build a small inference script
