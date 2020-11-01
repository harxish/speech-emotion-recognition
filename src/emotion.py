import os
import pickle

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from feature_extractor import load_data


X_train, X_test, y_train, y_test = load_data(test_size=0.25)
print("[+] Number of training samples:", X_train.shape[0])
print("[+] Number of testing samples:", X_test.shape[0])
print("[+] Number of features:", X_train.shape[1])
print("[+] Number of classes:", y_train)

model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive',  
    'max_iter': 500, 
}

model = MLPClassifier(**model_params)

print("[*] Training the model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))