import pandas as pd 
import numpy as np
import joblib 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import accuracy_score

# Charger les données
data = pd.read_csv(r"C:\Users\lenovo\Downloads\apro\static\diabetes.csv")

# Diviser les données en variables explicatives (features) et cible
x = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialiser les modèles
model1 = LogisticRegression()
model2 = SVC() 
model3 = RandomForestClassifier() 
model4 = GradientBoostingClassifier()

# Définir une fonction pour entraîner et évaluer les modèles
def train_and_evaluate(model):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    return model

# Entraîner et évaluer les modèles
model1 = train_and_evaluate(model1)
model2 = train_and_evaluate(model2)
model3 = train_and_evaluate(model3)
model4 = train_and_evaluate(model4)

# Enregistrer les modèles entraînés
joblib.dump(model1, 'model1_diabetes.pkl')
joblib.dump(model2, 'model2_diabetes.pkl')
joblib.dump(model3, 'model3_diabetes.pkl')
joblib.dump(model4, 'model4_diabetes.pkl')
