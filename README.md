## 🖋️ Signature Classification using CNNs (MobileNetV2 vs EfficientNetV2)

This project focuses on building and comparing two powerful convolutional neural network architectures — **MobileNetV2** and **EfficientNetV2-B0** — for **binary image classification**: detecting real vs forged handwritten signatures.


### 🧠 Objective

To classify signature images into two categories:
- **Real (1)**
- **Forged (0)**

Using transfer learning with pre-trained models to boost performance on a relatively small dataset.


### 📁 Dataset

- Source: [Kaggle Signature Dataset](https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures)  
- Structure:
  ```
  Dataset/
  ├── dataset1/
  │   ├── real/
  │   └── forge/
  ├── dataset2/
  │   └── ...
  ```

- Image Size: Resized to **256x256**
- Total Samples: 630 (after cleaning)

### 🔧 Preprocessing

- All images resized to `(256, 256)`
- Normalized pixel values to range [0, 1]
- Converted grayscale images to RGB
- Labels: 0 = forged, 1 = real

### 🔍 Models Used

#### ✅ MobileNetV2
- Lightweight and fast
- Final accuracy: **~90%**
- Frozen base + custom classification head

#### ✅ EfficientNetV2-B0
- Modern, more complex architecture
- Final accuracy: **~68%**
- Used as backbone with custom head

---

### 🧪 Results

| Model           | Validation Accuracy |
|-----------------|---------------------|
| MobileNetV2     | **90%**             |
| EfficientNetV2  | **68%**             |

### 🛠️ Tools & Libraries

- Python 3.x
- TensorFlow / Keras
- KerasCV
- NumPy, Matplotlib, PIL
- opencv
- tdqm

### 📈 Training Code Snippet

```python
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```


### 📌 How to Run

1. Clone the repo
2. Install required libraries: `pip install -r requirements.txt`
3. Run `main.ipynb` or the training script


### 🤔 Future Improvements

- Fine-tune EfficientNet further
- Add dropout and batch normalisation
- Try other pre-trained models (e.g. ResNet, Inception)
- Test on unseen signature datasets


### 📸 Sample Image

![Sample Signature](![00101001](https://github.com/user-attachments/assets/816784a1-4079-4783-9676-723d330db761)
)
