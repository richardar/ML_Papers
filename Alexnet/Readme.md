# 🧠 AlexNet Paper Summary

This repository contains concise notes, summaries, and essential resources related to the groundbreaking paper **“ImageNet Classification with Deep Convolutional Neural Networks”** by *Alex Krizhevsky, Ilya Sutskever,* and *Geoffrey Hinton*.

---

## 📌 Overview

AlexNet marked a turning point in the field of deep learning by decisively winning the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012**. It demonstrated that deep convolutional neural networks (CNNs) could significantly outperform traditional computer vision techniques in large-scale image classification tasks.

---

## 🚀 Key Contributions

- 🧱 Introduced a deep CNN with **8 layers** (5 convolutional + 3 fully connected).
- ⚡ Adopted **ReLU** activations, enabling faster and more effective training.
- 🛡️ Used **dropout** to combat overfitting.
- 🖥️ Leveraged **GPU acceleration** for training on large datasets.
- 🖼️ Applied **data augmentation** and **normalization** techniques to boost performance.

---

## 📄 Paper Details

- **Title:** *ImageNet Classification with Deep Convolutional Neural Networks*  
- **Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
- **Published:** 2012  
- **Link:** [Read the Paper (NIPS)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

---

## 🗂️ Repository Structure

| File              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `engine.py`       | Defines the AlexNet architecture and training logic.                        |
| `train.py`        | Script to train the model.                                                  |
| `cust_dataset.py` | Custom PyTorch `Dataset` class to load and preprocess data.                 |
| `main.py`         | Entry point script to start training.                                       |
| `main.ipynb`      | Initial notebook for experimentation and prototyping.                       |

---

## 🛠️ How to Use

```
python main.py
```

---

## 📘 License

This project is intended for **educational purposes only**.
