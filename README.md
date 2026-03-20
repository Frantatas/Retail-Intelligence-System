# 🛍️ Retail Intelligence System

A machine learning-based system that predicts retail sales and classifies transactions as HIGH or LOW using multiple models.

---

## 📌 Project Overview

This project uses a supermarket sales dataset to build a **Retail Intelligence System** that:

- Predicts total sales using **Linear Regression**
- Classifies transactions as HIGH or LOW using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Neural Network
- Provides a user-friendly interface using **Tkinter**

---

## 🤖 Machine Learning Models Used

| Model | Purpose |
|------|--------|
| Linear Regression | Predict sales amount |
| Logistic Regression | Classify HIGH/LOW sale |
| Decision Tree | Classification |
| Random Forest | Classification (best overall) |
| Neural Network | Classification |

---

## 📊 Dataset

- Source: Supermarket Sales Dataset (Kaggle)
- Rows: 1000
- Features include:
  - Quantity
  - Unit Price
  - Product Line
  - Payment Method
  - Customer Type
  - Gender
  - Date (converted to Month)

---

## 🖥️ How to Run the Project

### 1. Install dependencies
```bash
pip install pandas scikit-learn joblib
