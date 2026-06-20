# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# print("Loading dataset...")
# # df = pd.read_csv("smell_dataset.csv")
# df = pd.read_csv("final_balanced_dataset.csv")

# print("Dataset shape:", df.shape)

# X = df.drop("label", axis=1)
# y = df["label"]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA(n_components=20)   
# X_pca = pca.fit_transform(X_scaled)

# print("Reduced shape after PCA:", X_pca.shape)


# X_train, X_test, y_train, y_test = train_test_split(
#     X_pca, y, test_size=0.2, random_state=42
# )


# print("Training model...")
# model = RandomForestClassifier(n_estimators=150, random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)

# print("\n✅ Model Accuracy:", accuracy)
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:\n")
# print(confusion_matrix(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

df = pd.read_csv("final_balanced_dataset.csv")

print("Dataset shape:", df.shape)

# remove batch column
X = df.drop(["label","batch"], axis=1)
y = df["label"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=40)

X_pca = pca.fit_transform(X_scaled)

print("Reduced shape after PCA:", X_pca.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Accuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))


