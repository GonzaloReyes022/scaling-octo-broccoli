# Hands-On Machine Learning Experiments

Production-ready ML project structure for exercises from "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow".

## Project Structure

```
scaling-octo-broccoli/
├── configs/              # YAML configuration files
│   ├── default.yaml      # Local development config
│   └── colab.yaml        # Google Colab config
├── data/                 # Data directory (gitignored)
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned, transformed data
│   └── external/         # Third-party data sources
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Standalone scripts
├── src/                  # Source code
│   ├── data/             # Data loading modules
│   ├── preprocessing/    # Data transformation pipelines
│   ├── models/           # Model definitions
│   ├── training/         # Training utilities
│   └── utils/            # Helper functions
├── tests/                # Unit tests
├── pyproject.toml        # Project dependencies & config
└── README.md
```

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/scaling-octo-broccoli.git
cd scaling-octo-broccoli

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev,notebook]"
```

### Google Colab

See [Colab Integration](#google-colab-integration) section below.

## Quick Start

### MNIST Classification (Exercise 1 & 2)

```python
from src.data import MNISTLoader
from src.preprocessing import ImageAugmentor
from src.models import ClassifierFactory
from src.training import Trainer, HyperparameterSearcher

# Load data
loader = MNISTLoader("data/raw/mnist_784.csv.zip")
X_train, X_test, y_train, y_test = loader.stratified_split()

# Hyperparameter search
searcher = HyperparameterSearcher(
    model=ClassifierFactory.create("knn", use_defaults=False),
    param_grid={"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    cv=4,
)
searcher.grid_search(X_train, y_train)
print(f"Best params: {searcher.best_params}")  # {'n_neighbors': 3, 'weights': 'distance'}

# Data augmentation
augmentor = ImageAugmentor(image_shape=(28, 28))
X_aug, y_aug = augmentor.augment_with_shifts(X_train, y_train)

# Train with best model
trainer = Trainer(searcher.best_estimator)
trainer.fit(X_aug, y_aug)
results = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")  # ~97.3%
```

### Titanic Classification (Exercise 3)

```python
from src.data import TitanicLoader
from src.preprocessing import TitanicPreprocessor
from src.models import ClassifierFactory
from src.training import Trainer

# Load and preprocess
loader = TitanicLoader("data/raw")
X, y = loader.get_features_and_labels()

preprocessor = TitanicPreprocessor()
X_processed = preprocessor.fit_transform(X)

# Compare models
for model_name in ["random_forest", "svm", "knn"]:
    model = ClassifierFactory.create(model_name)
    trainer = Trainer(model)
    cv_results = trainer.cross_validate(X_processed, y, cv=10)
    print(f"{model_name}: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
```

## Configuration

Configurations are stored in `configs/` as YAML files:

```python
from src.utils import load_config

# Load default config
config = load_config("default")

# Load Colab-specific config
config = load_config("colab")
```

Environment variables can be referenced using `${VAR_NAME}` syntax in config files.

## Google Colab Integration

### Option 1: Mount Google Drive (Recommended)

This preserves your work between sessions:

```python
# In Colab notebook - Cell 1
from google.colab import drive
drive.mount('/content/drive')

# Clone repo to Drive (first time only)
%cd /content/drive/MyDrive
!git clone https://github.com/yourusername/scaling-octo-broccoli.git

# Navigate to project
%cd /content/drive/MyDrive/scaling-octo-broccoli

# Install dependencies
!pip install -e ".[notebook]"
```

### Option 2: Fresh Clone Each Session

```python
# In Colab notebook
!git clone https://github.com/yourusername/scaling-octo-broccoli.git
%cd scaling-octo-broccoli
!pip install -e ".[notebook]"
```

### Option 3: VSCode + Colab Extension

1. Install the **"Colab for VS Code"** extension in VS Code
2. Open a notebook in VS Code
3. Click "Connect" and select "Connect to Google Colab"
4. Your local code runs on Colab's GPU/RAM

**Setup cell for remote execution:**

```python
# Mount drive and setup project
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/scaling-octo-broccoli')

# Now you can import your modules
from src.data import MNISTLoader
from src.training import Trainer
```

## Exercises Implemented

| Exercise | Description | Module |
|----------|-------------|--------|
| Ch3 Ex1 | MNIST KNN Classifier (>97% accuracy) | `src/data/mnist_loader.py` |
| Ch3 Ex2 | Data Augmentation with Image Shifts | `src/preprocessing/image_augmentation.py` |
| Ch3 Ex3 | Titanic Survival Prediction | `src/data/titanic_loader.py`, `src/preprocessing/tabular_pipelines.py` |

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/

# Type checking
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) file.
