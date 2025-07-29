# Total-perspective-Vortex

A comprehensive Brain-Computer Interface system for processing and analyzing EEG (Electroencephalogram) data using machine learning techniques. This project implements advanced signal processing pipelines including Common Spatial Patterns (CSP), ANOVA-F feature selection, and Riemannian geometry-based classification.

## 🧠 Features

- **EEG Data Processing**: Load and process MNE-compatible EEG epoch files
- **Advanced Feature Extraction**: 
  - Common Spatial Patterns (CSP) for spatial filtering
  - ANOVA-F test for channel selection
  - Riemannian geometry-based covariance estimation
- **Machine Learning Pipeline**: Complete scikit-learn compatible pipeline with SVM classification
- **Docker Support**: Containerized environment for easy deployment
- **Jupyter Integration**: Development environment with Jupyter notebooks
- **Multi-subject Support**: Process data from multiple subjects (S001-S010)

## 📁 Project Structure

```
tpvcom/
├── app/
│   ├── features/
│   │   ├── CSP.py              # Common Spatial Patterns implementation
│   │   └── ANOVA_F.py          # ANOVA-F feature selection
│   ├── processing/
│   │   └── Processor.py        # Data processing utilities
│   ├── Visualizer/
│   │   └── Visualizer.py       # Data visualization tools
│   ├── mybci.py               # Main BCI application
│   ├── split_epochs.py        # Epoch splitting utilities
│   └── subjects.txt           # Subject list
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker container definition
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tpvcom
   ```

2. **Start the BCI application**
   ```bash
   docker-compose up bci-app
   ```

3. **For development with Jupyter**
   ```bash
   docker-compose --profile dev up jupyter
   ```
   Then access Jupyter at `http://localhost:8888`

### Local Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   python app/mybci.py <start_run> <end_run> <mode>
   ```

## 📊 Usage

### Command Line Interface

The main application supports two modes:

#### Training Mode
```bash
python app/mybci.py 1 8 train
```
- Processes runs 1-7 for training
- Saves trained model to `models/model.pkl`

#### Prediction Mode
```bash
python app/mybci.py 1 8 predict
```
- Loads trained model and makes predictions on test data
- Displays prediction accuracy and detailed results

### Data Organization

Expected directory structure:
```
data/
├── train/                    # Training data
│   ├── S001-epo.fif
│   ├── S002-epo.fif
│   └── ...
├── test/                     # Test data
│   ├── S001-epo.fif
│   ├── S002-epo.fif
│   └── ...
└── epochs/
    ├── train/               # Processed training epochs
    └── test/                # Processed test epochs
```

## 🔬 Technical Details

### Machine Learning Pipeline

The system implements a sophisticated pipeline:

1. **ANOVA-F Feature Selection**: Selects the most discriminative EEG channels
2. **Common Spatial Patterns (CSP)**: Extracts spatial features for better class separation
3. **Covariance Estimation**: Computes Riemannian covariance matrices
4. **Tangent Space Mapping**: Projects data to Riemannian tangent space
5. **SVM Classification**: Linear Support Vector Machine for final classification

### Key Components

- **CSP.py**: Implements Common Spatial Patterns algorithm for spatial filtering
- **ANOVA_F.py**: ANOVA-F test for channel selection and feature reduction
- **mybci.py**: Main application orchestrating the entire pipeline

## 🐳 Docker Configuration

### Services

- **bci-app**: Main BCI processing application
- **jupyter**: Development environment with Jupyter notebooks (dev profile)

### Volumes

- `eeg_data`: Persistent storage for EEG data
- `eeg_models`: Persistent storage for trained models
- `./data`: Local data directory mounted to container

## 📦 Dependencies

- **mne**: EEG/MEG data processing
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **joblib**: Model persistence
- **pyriemann**: Riemannian geometry for EEG
- **pathlib2**: Path manipulation utilities

## 🔧 Development

### Adding New Features

1. Create feature classes in `app/features/`
2. Implement scikit-learn compatible interface (BaseEstimator, TransformerMixin)
3. Integrate into pipeline in `mybci.py`

### Customizing the Pipeline

Modify the `create_pipeline()` function in `mybci.py`:

```python
def create_pipeline():
    return Pipeline([
        ('ANOVA_F', ANOVA_F(k=10)),           # Adjust k for channel selection
        ('CSP', CSP(n_components=6)),         # Adjust components
        ('Covariances', Covariances(estimator='lwf')),
        ('TangentSpace', TangentSpace(metric='riemann')),
        ('SVC', SVC(kernel='linear', C=0.1, probability=True)),
    ])
```
