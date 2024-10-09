import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 

# Charger les données
data_path = r"C:\Users\lenovo\Downloads\backend\maladie_observations.csv"
data = pd.read_csv(data_path)

data

"""## *Exploration data*"""

print( data.head())

data.info()
# on a choisi cette methode puisque elle nous donne plus de precision dans les trois modeles de prediction qu'on a choisi
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
objet=IterativeImputer(max_iter=100,random_state=42)
data=objet.fit_transform(data)

type(data)

# changer le type de data
data= pd.DataFrame(data)
type(data)

from sklearn.model_selection import train_test_split
x=data.drop(5, axis=1)
y=data[5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""*L'optimisation du Model1*"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Assuming you have x_train and y_train defined

model = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 32), 'metric': ['euclidean', 'manhattan']}
grid = GridSearchCV(model, param_grid, cv=3, error_score='raise')

try:
    grid.fit(x_train, y_train)
    print("Best Score:", grid.best_score_)
    print("Best Parameters:", grid.best_params_)
except Exception as e:
    print("An error occurred during grid search:", e)

"""*L'optimisation de model 2*"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score,r2_score

model=KNeighborsClassifier(metric= 'manhattan',n_neighbors=11)
def cal(model):
   model.fit(x_train, y_train)
   y_pred = model.predict(x_test)
   accuracy=accuracy_score(y_test,y_pred)
   print("accuracy=",accuracy)

cal(model)

#construction du modele RandomForestClassifier puisque c'est le model qui nous donne plus de precision
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print(y_pred,y_test)

joblib.dump(model, 'corona.pkl')