# 🏠 Multimodal Housing Price Prediction (Images + Tabular Data)

## 📌 Project Overview
This project implements a **Multimodal Machine Learning** approach to predict **housing prices** using both **structured/tabular data** and **house images**.  
By combining **CNN-based image features** with **tabular features**, the model achieves improved prediction accuracy compared to using a single data modality.

---

## 🎯 Objective
- Extract image features using **Convolutional Neural Networks (CNNs)**  
- Combine image and tabular features for **joint learning**
- Train a **regression model** to predict house prices
- Evaluate model performance using **MAE** and **RMSE**

---

## 🧠 Skills Gained
- Multimodal Machine Learning  
- Deep Learning with **CNNs (ResNet18)**  
- Feature Fusion (Image + Tabular Data)  
- Regression Modeling and Evaluation  
- PyTorch-based Model Training

---

## 📂 Dataset
The project uses a **Housing Sales Dataset** combined with corresponding **house images**.

You can use:
- 🧾 **Tabular Data**: [Kaggle - House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- 🏡 **Image Data**: Publicly available house images or custom image dataset matching the tabular entries.

---

## ⚙️ Preprocessing
1. **Tabular Data**:
   - Missing value handling  
   - Feature scaling using `StandardScaler`  
   - Train-validation-test split  

2. **Image Data**:
   - Resized to `(224x224)`  
   - Normalized for ResNet input

3. **Target Scaling**:
   ```python
   from sklearn.preprocessing import StandardScaler
   price_scaler = StandardScaler()
   train_df['price_scaled'] = price_scaler.fit_transform(train_df[['price']])
   val_df['price_scaled'] = price_scaler.transform(val_df[['price']])
   test_df['price_scaled'] = price_scaler.transform(test_df[['price']])





## 🧱 Model Architecture

The model consists of:

- 🧠 **CNN Encoder**: `ResNet18` pretrained on **ImageNet**  
  Extracts visual features from house images.

- 📊 **Tabular Network**: Fully connected layers  
  Processes numerical and categorical house attributes.

- 🔗 **Fusion Layer**: Combines CNN + Tabular features for regression  
  Merges learned representations and outputs the final price prediction.

---

## 🚀 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Weight Decay** | 1e-5 |
| **Epochs** | 30 |
| **Batch Size** | 32 |

Regularization techniques like **Dropout**, **Weight Decay**, and **Learning Rate Scheduling** are used to **reduce overfitting** and improve **generalization**.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test MAE** | 47,097.14 |
| **Test RMSE** | 59,393.88 |

---

## 📈 Error Analysis

Below is the **Prediction Error Distribution**, showing how predicted prices deviate from actual values.

> The distribution is centered near zero, indicating **balanced performance** and **minimal bias**.

---

## 🧪 Evaluation

- **Mean Absolute Error (MAE)** → measures the average deviation of predictions.  
- **Root Mean Squared Error (RMSE)** → penalizes larger errors more heavily.  
- Visualization helps identify **bias** or **variance** in model predictions.

---

## 💡 Insights

- 🏠 CNN captures **visual cues** such as house size, architecture, and surroundings.  
- 📋 Tabular features (e.g., `sqft`, `bedrooms`, `zipcode`) provide **structured context**.  
- 🔗 Combined model outperforms **single-modality** models (image-only or tabular-only).

---

## 🛠️ Tech Stack

- 🐍 Python 3.x  
- 🔥 PyTorch  
- 📈 scikit-learn  
- 📊 matplotlib / seaborn  
- 📘 NumPy / Pandas  

---

## 📘 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/multimodal-housing-price-prediction.git
   cd multimodal-housing-price-prediction
