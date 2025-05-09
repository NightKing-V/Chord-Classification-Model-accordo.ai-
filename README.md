# Chord-Classification-Model-accordo.ai

## Deep Learning for Automated Chord Recognition

This repository contains the research and implementation of deep learning models for real-time chord analysis and recognition, developed as part of the Accordo.ai project.

## Project Overview

Accordo.ai is a system designed to provide real-time chord recognition from audio input. The project explores several deep learning approaches to accurately identify musical chords from audio signals, breaking down chord components into root notes, bass notes, triads, and fourth notes.

## Models

The research developed and evaluated three main models:

### M1: Chroma-based Single Output Layer Model
- **Dataset**: McGill Billboard dataset
- **Features**: 12 chroma features
- **Architecture**: Bi-LSTM with single output layer
- **Purpose**: Initial research and experimentation

### M2: Chroma-based Multi-Output Layer Model
- **Dataset**: Isophonics dataset (The Beatles)
- **Features**: 12 chroma features
- **Architecture**: Bi-LSTM with four output layers (root, bass, triad, fourth)
- **Data Augmentation**: Pitch shifting (±3 semitones)

### M3: CQT-based Multi-Output Layer Model
- **Dataset**: Isophonics dataset (The Beatles)
- **Features**: 192 CQT features (24 bins per octave for 8 octaves)
- **Architecture**: Bi-LSTM with four output layers (root, bass, triad, fourth)
- **Data Augmentation**: Pitch shifting (±5 semitones), Gaussian noise
- **Additional Techniques**: Custom frequency weighting, batch normalization

## Data Processing

The project implements several data processing techniques:
- **Feature Extraction**: Using librosa for chroma and CQT features
- **Chord Standardization**: Handling enharmonic equivalents and inconsistent labeling
- **Stratified Splitting**: Ensuring rare chords are represented in all data splits
- **Data Augmentation**: Expanding dataset size through pitch shifting and noise
- **Feature Engineering**: Converting chord labels to vectors and one-hot encodings

## Model Performance

M3 demonstrated the highest overall performance among the three models, with better accuracy when evaluated against a test set. However, challenges remain with complex audio signals and rare chord types.

## Limitations and Future Work

- **Data Limitations**: More varied training data across genres is needed
- **Computational Resources**: Higher computing power for more complex models
- **Chord Complexity**: Difficulty in recognizing complex or rare chord combinations
- **Multi-Output Sensitivity**: Small errors in one output affect the entire chord prediction

Future improvements would focus on:
1. Expanding the training dataset with more diverse musical genres
2. Implementing more robust post-processing algorithms
3. Exploring alternative neural network architectures
4. Optimizing for real-time performance

## Dependencies

Main dependencies include:
- TensorFlow/Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- h5py

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/accordo-ai.git
cd accordo-ai

# Create and activate a virtual environment (optional)
docker-composer up --build -d

```

## Citation

If you use this code or research in your work, please cite:

@misc{accordoai2025,
  author = {Your Name},
  title = {Accordo.ai: Deep Learning for Automated Chord Recognition},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/accordo-ai}
}

## License

[MIT License](LICENSE)
