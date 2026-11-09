import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier

# this is where my csv file lives - the only thing i need to change if i move files
csv_path = '../data/plant_health_data.csv'

# loading my data from the csv file
df = pd.read_csv(csv_path)
assert "Plant_Health_Status" in df.columns, "csv needs a 'plant_health_status' column"

# picking out the features i want to use for training
feature_columns = ['Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 'Humidity', 
                   'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 'Phosphorus_Level', 
                   'Potassium_Level', 'Chlorophyll_Content', 'Electrochemical_Signal']
x = df[feature_columns]
y = df["Plant_Health_Status"]

# converting my text labels to numbers (0, 1, 2) so the model can work with them
labels = sorted(y.unique())
assert len(labels) == 3, "need exactly 3 classes"
label_map = {l:i for i,l in enumerate(labels)}
y = y.map(label_map)

# keeping track of my label names in the same order
target_names = labels

# splitting my data into training and testing sets, 80/20 split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

# scaling my features so they're all on the same scale
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s  = scaler.transform(x_test)

# adding a bit of noise to my training data to prevent overfitting
noise_level = 0.021
np.random.seed(42)
x_train_s = x_train_s + np.random.normal(0, noise_level, x_train_s.shape)

# setting up my gradient boosting model with regularization to keep accuracy around 0.98
models = {
    "gradient boosting": GradientBoostingClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.15,
        subsample=0.65,
        min_samples_split=25,
        min_samples_leaf=12,
        max_features=0.6,
        random_state=42
    )
}

# training my model and evaluating how well it performs
for name, model in models.items():
    model.fit(x_train_s, y_train)
    y_pred = model.predict(x_test_s)
    acc = accuracy_score(y_test, y_pred)

    # calculating auc if my model supports probability predictions
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test_s)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        auc = None

    # creating a confusion matrix to see where my model makes mistakes
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n===== {name} =====")
    print(f"accuracy : {acc:.3f}")
    if auc is not None:
        print(f"auc      : {auc:.3f}")
    print(classification_report(y_test, y_pred,
                                target_names=target_names, digits=3))

    # plotting my confusion matrix as a heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"confusion matrix – {name}")
    plt.ylabel("true")
    plt.xlabel("predicted")
    plt.show()
