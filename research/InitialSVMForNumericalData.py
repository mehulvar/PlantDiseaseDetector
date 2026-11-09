import pandas as pd

# Load your dataset
df = pd.read_csv('../data/plant_health_data.csv')

# Drop non-numeric or irrelevant columns
df = df.drop(columns=['Timestamp', 'Plant_ID'])

from sklearn.preprocessing import LabelEncoder

# Encode 'Plant_Health_Status' to numeric labels
label_encoder = LabelEncoder()
df['Plant_Health_Status'] = label_encoder.fit_transform(df['Plant_Health_Status'])

# Separate features and target
X = df.drop(columns=['Plant_Health_Status'])
y = df['Plant_Health_Status']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
