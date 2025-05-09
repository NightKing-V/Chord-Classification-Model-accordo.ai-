# 🎼 Chord-Classification-Model-Accordo.ai

## 🤖 Deep Learning for Automated Chord Recognition

This repository contains the research and implementation of deep learning models for real-time chord analysis and recognition, developed as part of the **Accordo.ai** project.

---

## 📌 Project Overview

**Accordo.ai** is a system designed to provide real-time chord recognition from audio input. The project explores several deep learning approaches to accurately identify musical chords from audio signals, breaking down chord components into **root notes**, **bass notes**, **triads**, and **fourth notes**.

---

## 🧠 Models

Three main models were developed and evaluated:

### 🎶 M1: Chroma-based Single Output Layer Model
- 📚 **Dataset**: McGill Billboard
- 🎛 **Features**: 12 chroma features
- 🧩 **Architecture**: Bi-LSTM with single output layer
- 🧪 **Purpose**: Initial experimentation

### 🎶 M2: Chroma-based Multi-Output Layer Model
- 📚 **Dataset**: Isophonics (The Beatles)
- 🎛 **Features**: 12 chroma features
- 🧩 **Architecture**: Bi-LSTM with four output layers (root, bass, triad, fourth)
- 🧬 **Augmentation**: Pitch shifting (±3 semitones)

### 🎶 M3: CQT-based Multi-Output Layer Model
- 📚 **Dataset**: Isophonics (The Beatles)
- 🎛 **Features**: 192 CQT features (24 bins/octave × 8 octaves)
- 🧩 **Architecture**: Bi-LSTM with four output layers (root, bass, triad, fourth)
- 🧬 **Augmentation**: Pitch shifting (±5 semitones), Gaussian noise
- ⚙️ **Extras**: Custom frequency weighting, batch normalization

---

## 🗃️ Datasets

```
Burgoyne, J.A., Wild, J., & Fujinaga, I. (2011). An expert ground-truth set for audio 
chord recognition and music analysis. Proceedings of the 12th International 
Society for Music Information Retrieval Conference (ISMIR), 633–638. 
(McGill Billboard dataset. https://www.kaggle.com/datasets/jacobvs/mcgill
billboard) 
 
Harte, C., Sandler, M., Abdallah, S., & Gómez, E. (2005). Symbolic representation 
of musical chords: A proposed syntax for text annotations. Proceedings of the 6th 
International Conference on Music Information Retrieval (ISMIR), 66–71. 
(Isophonics dataset. http://isophonics.net/content/reference-annotations
beatles) 

```
## ⚙️ Data Processing Pipeline

- 🔍 **Feature Extraction**: Librosa (chroma & CQT)
- 🎼 **Chord Standardization**: Enharmonic correction, consistent labeling
- 🔁 **Stratified Splitting**: Balanced training/validation/test splits
- 🧬 **Augmentation**: Pitch shifting, noise injection
- 🧠 **Vectorization**: One-hot encoding of chord components

---

## 📈 Model Performance

Model **M3** showed the highest accuracy in testing and generalization. However, recognition of rare chords and noisy signals remains a challenge for all models.

---

## ⚠️ Limitations & 🚀 Future Work

- 🔎 Need for more diverse genre datasets
- 💻 High computational requirements for training
- 🎶 Difficulty with complex or ambiguous chords
- ⚖️ Sensitivity in multi-output predictions

**Planned improvements:**
1. Expanding dataset coverage
2. Enhancing post-processing logic
3. Exploring new model architectures (e.g., Transformers)
4. Real-time inference optimization

---

## 🧰 Dependencies

- TensorFlow / Keras  
- Librosa  
- NumPy  
- Pandas  
- Matplotlib  
- h5py  

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/NightKing-V/Chord-Classification-Model-accordo.ai-.git

# Start services using Docker Compose
docker-compose up --build -d

```
## 🔖 Citation

If you use this code or research in your work, please cite:
```
@misc{accordoai2025,
  author       = {Robalge Valenteno Lenora},
  title        = {Accordo.ai: Deep Learning for Automated Chord Recognition},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/NightKing-V/Chord-Classification-Model-accordo.ai-.git}
}
```
## 🙌 Credits
```
Aslanidis, T. A. (2020). Deep Learning in Audio Chord Estimation (Bachelor's 
thesis). Department of Informatics and Telecommunications, National and 
Kapodistrian University of Athens.  (https://github.com/taslanidis/Audio-Chord
Recognition) 

Harte, C. A. (2010). "Towards automatic extraction of harmony information from 
music signals." Proceedings of the 11th International Conference on Digital Audio 
Effects (DAFx-08).
```

## 📱 Implmentations
```
Music Analysis System (AndroidApp + Backend)
https://github.com/NightKing-V/accordo.ai
```

## 👨‍🎓 About the Author & Project Motivation

This research project was developed by Robalge Valenteno Lenora as part of the final year requirements for the BSc (Hons) Computer Science degree at the University of Plymouth.

Supervised by Dr. Mohamed Shafraz, the project explores how deep learning techniques can be applied to automate musical chord recognition and beat analysis — a task traditionally done manually or using rule-based systems. The motivation behind Accordo.ai stems from a passion for music and artificial intelligence, with a desire to bridge both fields to build intelligent tools for musicians, learners, and researchers.

This work not only contributes to the field of music information retrieval (MIR) but also demonstrates real-world application of modern AI engineering practices, including microservice architecture, containerization, and scalable machine learning deployment.

## 📜 License

[MIT License](LICENSE)
