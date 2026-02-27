import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------- LOAD DATA ----------------
df = pd.read_csv("heart.csv")

# encode categorical columns
data = df.copy()
for col in data.select_dtypes(include="object"):
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# ---------------- GUI WINDOW ----------------
root = tk.Tk()
root.title("Heart Disease Clinical Dashboard")
root.geometry("700x600")
root.configure(bg="#f4f6f7")

# ---------------- HEADER ----------------
title = tk.Label(root,
                 text="Heart Disease Analysis System",
                 font=("Helvetica",18,"bold"),
                 bg="#2c3e50",
                 fg="white",
                 pady=15)
title.pack(fill="x")

# ---------------- INFO ----------------
info = tk.Label(root,
                text=f"Model Accuracy: {round(accuracy*100,2)}%",
                font=("Arial",14),
                bg="#f4f6f7",
                fg="#2c3e50")
info.pack(pady=10)

# ---------------- GRAPH FUNCTIONS ----------------
def age_vs_hr():
    plt.figure()
    df.groupby("Age")["MaxHR"].mean().plot()
    plt.title("Average Heart Rate by Age")
    plt.xlabel("Age")
    plt.ylabel("Average MaxHR")
    plt.grid(True)
    plt.show()

def cholesterol_trend():
    plt.figure()
    df.groupby("Age")["Cholesterol"].mean().plot()
    plt.title("Cholesterol Trend by Age")
    plt.xlabel("Age")
    plt.ylabel("Cholesterol")
    plt.grid(True)
    plt.show()

def bp_trend():
    plt.figure()
    df.groupby("Age")["RestingBP"].mean().plot()
    plt.title("Blood Pressure Trend by Age")
    plt.xlabel("Age")
    plt.ylabel("Resting BP")
    plt.grid(True)
    plt.show()

def feature_importance():
    imp = model.feature_importances_
    cols = X.columns
    plt.figure()
    plt.plot(cols, imp, marker="o")
    plt.xticks(rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# ---------------- BUTTONS ----------------
frame = tk.Frame(root, bg="#f4f6f7")
frame.pack(pady=20)

buttons = [
    ("Heart Rate Trend", age_vs_hr),
    ("Cholesterol Trend", cholesterol_trend),
    ("Blood Pressure Trend", bp_trend),
    ("Feature Importance", feature_importance)
]

for text, func in buttons:
    ttk.Button(frame, text=text, command=func, width=25).pack(pady=8)

# ---------------- PREDICTION SECTION ----------------
predict_label = tk.Label(root,
                         text="Predict Risk from Sample Patient",
                         font=("Arial",14,"bold"),
                         bg="#f4f6f7")
predict_label.pack(pady=15)

def predict():
    sample = X.iloc[0].values.reshape(1,-1)
    result = model.predict(sample)[0]

    if result == 1:
        msg.config(text="⚠ High Risk Detected", fg="red")
    else:
        msg.config(text="✓ Low Risk", fg="green")

ttk.Button(root, text="Run Prediction", command=predict).pack()

msg = tk.Label(root, text="", font=("Arial",14), bg="#f4f6f7")
msg.pack(pady=10)

# ---------------- FOOTER ----------------
footer = tk.Label(root,
                  text="Clinical Decision Support Prototype",
                  bg="#2c3e50",
                  fg="white",
                  pady=10)
footer.pack(side="bottom", fill="x")

root.mainloop()