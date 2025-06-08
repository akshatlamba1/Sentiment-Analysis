# Sentiment Analysis on Rotten Tomatoes Reviews

This project uses deep learning to predict the sentiment (positive or negative) of movie reviews from the Rotten Tomatoes dataset. It combines **text vectorization**, **Convolutional Neural Networks (CNNs)**, and **Long Short-Term Memory (LSTM)** layers to build a powerful sentiment analysis model.

---

## Features

✅ Preprocessing of textual data
✅ Text vectorization with `TextVectorization` layer
✅ Use of CNN layers to capture local patterns
✅ LSTM layers for sequential dependencies
✅ Dropout and L2 regularization to prevent overfitting
✅ Model evaluation with metrics like accuracy, precision, recall, F1-score, and confusion matrix

---

## Project Structure

```
📁 project_root/
├── data/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── notebooks/
│   ├── Preprocessing.ipynb
│   ├── Model_Building.ipynb
│   ├── Model_Training.ipynb
│   └── Evaluation.ipynb
├── vectorizer_model.h5
├── model.h5
└── README.md
```

---

## Key Components

### 📚 Preprocessing

* Lowercasing, punctuation removal, and tokenization
* Vectorization of text into numerical format
* Split into training, validation, and testing sets

### ⚙️ Model Architecture

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
])
```

---

### 🧪 Model Training

* Loss function: `binary_crossentropy`
* Optimizer: `adam`
* Batch size: 32
* Epochs: 10
* Early stopping to prevent overfitting

---

### 📈 Evaluation

* **Accuracy**: Final training & validation accuracy \~84%
* **Loss**: Final training & validation loss \~36
* **Confusion Matrix**, **Precision**, **Recall**, and **F1-score** computed using `sklearn.metrics`

---

### 🚀 Running the Project

1. Clone this repository

   ```bash
   git clone https://github.com/yourusername/rotten-tomatoes-sentiment.git
   cd rotten-tomatoes-sentiment
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in order:

   * `Preprocessing.ipynb`
   * `Model_Building.ipynb`
   * `Model_Training.ipynb`
   * `Evaluation.ipynb`

4. Evaluate performance on the test dataset using:

   ```python
   y_pred = (model.predict(x_test) > 0.5).astype(int)
   ```

---

## ⚡ Key Learnings

* Importance of vectorization and regularization in text-based models
* Balancing overfitting and underfitting using dropout and L2 regularization
* Using CNNs for capturing local patterns and LSTMs for long-range dependencies in text

---

## 🤝 Contributing

Feel free to open issues or create pull requests if you’d like to contribute to this project!

---

## 📄 License

This project is licensed under the MIT License.

---
