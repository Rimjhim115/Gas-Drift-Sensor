import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models

def build_resnet(input_shape, n_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    def res_block(x, filters, kernel=3):
        shortcut = layers.Conv1D(filters, 1, padding='same')(x)
        x = layers.Conv1D(filters, kernel, padding='same', activation='relu')(x)
        x = layers.Conv1D(filters, kernel, padding='same')(x)
        x = layers.Add()([x, shortcut])
        return layers.Activation('relu')(x)

    x = res_block(x, 32)
    x = res_block(x, 64)
    x = res_block(x, 128)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs)

print("Loading dataset...")
df = pd.read_csv("final_balanced_dataset.csv")
batches = sorted(df["batch"].unique())
train_batch = batches[0]
test_batches = batches[1:]

train_df = df[df["batch"] == train_batch]
X_train = train_df.drop(["label", "batch"], axis=1).values
y_train = train_df["label"].values

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
n_classes = len(le.classes_)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = build_resnet((X_train_scaled.shape[1],), n_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train_enc, epochs=50, batch_size=32, verbose=0)

print("\nTask     | Accuracy")
print("-" * 22)
accs = []
for b in test_batches:
    b_df = df[df["batch"] == b]
    X_test = scaler.transform(b_df.drop(["label", "batch"], axis=1).values)
    y_test = le.transform(b_df["label"].values)
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, preds)
    accs.append(acc)
    print(f"1-{b:<6}  | {acc:.4f}")
print("-" * 22)
print(f"Average  | {np.mean(accs):.4f}")
