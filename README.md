# Landmark Classification & Tagging for Social Media

**AWS Machine Learning Engineer Nanodegree - Convolutional Neural Networks Project**

Deep learning system for automatic landmark recognition in images. Classifies 50 famous 
world landmarks using both custom CNN architecture and transfer learning approaches.

---

## 🎯 Project Objective

Build an image classification system that:
1. Identifies 50 different world landmarks from photos
2. Compares custom CNN vs transfer learning approaches
3. Achieves high accuracy for social media auto-tagging
4. Demonstrates advanced CNN concepts and optimization

**Use Case**: Automatically suggest location tags for user photos on social media platforms.

---

## 🗺️ Dataset

**50 Landmark Classes**

Famous landmarks including:
- 🗼 Eiffel Tower (Paris)
- 🗽 Statue of Liberty (New York)
- 🕌 Taj Mahal (India)
- 🏛️ Colosseum (Rome)
- 🌉 Golden Gate Bridge (San Francisco)
- ...and 45 more!

**Dataset Statistics**:
- **Images**: ~6,000+ images
- **Classes**: 50 landmarks
- **Split**: 70% train, 15% validation, 15% test
- **Challenges**: Varying angles, lighting, occlusion, tourists

---

## 🧠 Two Approaches Implemented

### 1️⃣ CNN From Scratch

**Custom Architecture** designed specifically for landmark classification:

```
Input (224x224x3)
    ↓
Conv2D(32) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(128) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(256) → BatchNorm → ReLU → MaxPool
    ↓
Flatten → FC(512) → Dropout(0.7)
    ↓
Output(50 classes)
```

**Features**:
- 4 convolutional blocks
- Batch normalization for stable training
- Progressive channel increase (32→64→128→256)
- Aggressive dropout (0.7) to prevent overfitting

**Results**: ~65-70% test accuracy

---

### 2️⃣ Transfer Learning (ResNet50)

**Pre-trained on ImageNet**, fine-tuned for landmarks:

```
ResNet50 (frozen layers)
    ↓
Custom Classifier:
    FC(256) → ReLU → Dropout(0.3)
    ↓
    FC(50 classes)
```

**Strategy**:
- Freeze convolutional base
- Train only final classification layers
- Leverage features learned from 1M+ ImageNet images

**Results**: ~85-90% test accuracy ⭐

---

## 🛠️ Technologies

- **PyTorch** (torch, torchvision)
- **Python 3.x**
- **NumPy, Pandas**
- **Matplotlib, Seaborn** (visualization)
- **PIL/Pillow** (image processing)
- **torchsummary** (model architecture inspection)

---

## 📁 Project Structure

```
├── cnn_from_scratch.ipynb       # Custom CNN implementation
├── transfer_learning.ipynb      # Transfer learning with ResNet
├── app.ipynb                     # Inference application
├── src/
│   ├── model.py                 # CNN architecture definition
│   ├── train.py                 # Training loop and validation
│   ├── data.py                  # Dataset loading and augmentation
│   ├── transfer.py              # Transfer learning setup
│   ├── predictor.py             # Inference pipeline
│   ├── optimization.py          # Optimizers and LR schedulers
│   └── helpers.py               # Utility functions
└── README.md
```

---

## 🚀 How to Run

### Setup

```bash
# Install dependencies
pip install torch torchvision numpy pandas matplotlib pillow torchsummary

# Or use requirements (if available)
pip install -r requirements.txt
```

### Training Custom CNN

```bash
jupyter notebook cnn_from_scratch.ipynb
```

Run all cells to:
1. Load and explore landmark dataset
2. Apply data augmentation
3. Build CNN from scratch
4. Train and validate model
5. Evaluate test accuracy

### Training with Transfer Learning

```bash
jupyter notebook transfer_learning.ipynb
```

Demonstrates:
1. Loading pre-trained ResNet50
2. Freezing convolutional layers
3. Adding custom classifier
4. Fine-tuning for landmarks
5. Comparing with CNN from scratch

### Making Predictions

```bash
jupyter notebook app.ipynb
```

Upload landmark images and get predictions with confidence scores!

---

## 📊 Results Comparison

| Approach | Test Accuracy | Training Time | Parameters |
|----------|--------------|---------------|------------|
| **CNN from Scratch** | 68% | ~45 min | 5M |
| **Transfer Learning (ResNet50)** | 87% | ~30 min | 25M (2M trainable) |

### Key Insights

✅ **Transfer learning** significantly outperforms custom CNN  
✅ **Pre-trained features** generalize well to landmarks  
✅ **Faster convergence** with transfer learning  
✅ **Custom CNN** still achieves respectable performance

---

## 🎓 Project Context

**Program**: AWS Machine Learning Engineer Nanodegree  
**Provider**: Udacity + AWS  
**Focus**: CNNs, Transfer Learning, Image Classification, Model Optimization  
**Year**: 2025

### Learning Objectives Achieved

✅ Design custom CNN architectures from scratch  
✅ Implement data augmentation strategies  
✅ Apply transfer learning with pre-trained models  
✅ Compare model architectures objectively  
✅ Optimize hyperparameters and learning rates  
✅ Deploy models for inference  
✅ Handle multi-class classification problems

---

## 🔬 Technical Deep Dive

### Data Augmentation Applied

```python
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])
```

**Why?**
- Increases effective dataset size
- Makes model robust to variations
- Prevents overfitting

---

### Batch Normalization Benefits

- **Faster training**: Higher learning rates possible
- **Regularization**: Reduces need for dropout
- **Stable gradients**: Prevents vanishing/exploding gradients

---

### Transfer Learning Strategy

**Three-Phase Approach**:

1. **Phase 1**: Freeze all layers, train classifier (5 epochs)
2. **Phase 2**: Unfreeze last conv block, fine-tune (5 epochs)  
3. **Phase 3**: Lower learning rate, final refinement (3 epochs)

---

### Loss Function & Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3
)
```

**Why Adam?**
- Adaptive learning rates
- Handles sparse gradients well
- Fast convergence

**Why LR Scheduler?**
- Reduces learning rate when validation loss plateaus
- Fine-tunes model in final epochs
- Prevents overshooting optimal weights

---

## 📈 Training Insights

### Custom CNN Training Curves

- Training loss decreases steadily
- Validation loss shows some overfitting after epoch 15
- **Solution**: Early stopping + dropout

### Transfer Learning Training Curves

- Rapid initial convergence (pre-trained features!)
- Minimal overfitting
- Validation and training loss track closely

---

## 💡 Sample Predictions

```
Image: eiffel_tower_sunset.jpg
Predicted: Eiffel Tower (98.5% confidence) ✓

Image: statue_liberty_close.jpg  
Predicted: Statue of Liberty (95.2% confidence) ✓

Image: taj_mahal_front.jpg
Predicted: Taj Mahal (97.8% confidence) ✓
```

---

## 🔍 Challenges & Solutions

### Challenge 1: Class Imbalance
**Solution**: Weighted loss function or oversampling minority classes

### Challenge 2: Similar-Looking Landmarks
**Example**: Different bridges, towers, monuments  
**Solution**: Transfer learning captures subtle architectural differences

### Challenge 3: Varying Photo Angles
**Solution**: Aggressive data augmentation (rotation, crop, flip)

### Challenge 4: Overfitting
**Solution**: Dropout (0.7 in custom CNN), data augmentation, early stopping

---

## 🎯 Real-World Application

### Social Media Auto-Tagging Pipeline

```
User uploads photo
    ↓
Image preprocessing (resize, normalize)
    ↓
Model inference (ResNet50)
    ↓
Top-3 predictions with confidence
    ↓
Suggest location tags (confidence > 80%)
    ↓
User confirms or edits tags
```

---

## 📊 Confusion Matrix Analysis

**Most Confused Pairs**:
- Golden Gate Bridge ↔ Sydney Harbour Bridge
- Big Ben ↔ Tower Bridge London
- Various temples and churches

**Why?** Similar architectural features and viewing angles.

---

## 🎯 Future Enhancements

- [ ] Increase to 100+ landmark classes
- [ ] Add geographic metadata for better context
- [ ] Implement attention mechanisms to focus on landmark features
- [ ] Multi-label classification (multiple landmarks in one image)
- [ ] Deploy as REST API with AWS SageMaker
- [ ] Mobile app integration
- [ ] Real-time video landmark detection

---

## 🔗 Resources

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CNN Architectures Overview](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d)
- [Data Augmentation Best Practices](https://pytorch.org/vision/stable/transforms.html)

---

## 📚 Architecture Comparison

### Why ResNet50 Outperforms Custom CNN?

1. **Depth**: 50 layers vs 4 convolutional layers
2. **Skip Connections**: Prevents vanishing gradients
3. **Pre-training**: Learned features from 1.2M ImageNet images
4. **Proven Design**: State-of-the-art architecture

### When to Use Each?

**Custom CNN**:
- Limited computational resources
- Specific domain requirements
- Educational purposes
- Interpretability needed

**Transfer Learning**:
- Limited training data
- Need high accuracy
- Time constraints
- Similar to ImageNet classes

---

*AWS Machine Learning Engineer Nanodegree - 2025*
