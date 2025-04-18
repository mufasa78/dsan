# Dual Stream Attention Network (DSAN) for Facial Expression Recognition

This project is a PyTorch implementation of the Dual Stream Attention Network (DSAN) for robust facial expression recognition in the wild.

## About DSAN

The Dual Stream Attention Network (DSAN) is designed for robust facial expression recognition in challenging real-world conditions. It consists of two main components:

1. **Global Feature Element-based Attention Network (GFE-AN)**: Applies sparse attention to selectively emphasize informative feature elements while suppressing those unrelated to facial expression.

2. **Multi-Feature Fusion-based Attention Network (MFF-AN)**: Extracts rich semantic information from different representation sub-spaces to make the network insensitive to occlusion and pose variation.

This architecture helps address challenges like facial occlusions and head pose variations that are common in real-world images.

## Features

- Complete implementation of the DSAN architecture
- Training pipeline with data augmentation and customizable hyperparameters
- Evaluation module for model assessment
- Visualization tools for attention weights and results
- Web interface with both Flask and Streamlit options
- Demo mode that works without requiring a pre-trained model

## Interface Options

This project provides two different user interfaces:

### 1. Flask Web Interface (Default)

The Flask web interface provides a traditional web application for facial expression recognition.

**To run the Flask interface:**
```bash
# Option 1: Using the provided script
./run.sh

# Option 2: Directly using gunicorn
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

Access the interface at: http://localhost:5000

### 2. Streamlit Interactive Interface

Streamlit provides a more interactive and data-focused interface for the same functionality.

**To run the Streamlit interface:**
```bash
# Option 1: Using the provided script
./run_streamlit.sh

# Option 2: Directly using streamlit
streamlit run streamlit_app.py
```

Access the interface at: http://localhost:8501

## System Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA-capable GPU (recommended for training)

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Training

To train the DSAN model on your dataset:

```bash
python train.py --data_dir /path/to/data --train_label train.txt --val_label val.txt
```

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --data_dir /path/to/data --val_label val.txt --checkpoint /path/to/checkpoint.pth
```

## Visualization

To visualize attention weights on a specific image:

```bash
python visualization.py --image_path /path/to/image.jpg --checkpoint /path/to/checkpoint.pth
```

## Demo Mode

Both interfaces support a demo mode that works without requiring a trained model. This is useful for testing the UI and functionality when you don't have a trained model available.

## License

This project is available under the MIT License.

## Acknowledgements

This implementation is based on the paper [Dual Stream Attention Network for Robust Facial Expression Recognition in the Wild].