from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import uuid

app = Flask(__name__)

# Define the upload folder directory
UPLOAD_FOLDER = 'C:\\Users\\lenovo\\Downloads\\apro\\static\\image predi'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the COVID-19 model
model_covid = joblib.load('corona.pkl')

# Load diabetes prediction models
model_diabetes1 = joblib.load('model1_diabetes.pkl')
model_diabetes2 = joblib.load('model2_diabetes.pkl')
model_diabetes3 = joblib.load('model3_diabetes.pkl')
model_diabetes4 = joblib.load('model4_diabetes.pkl')

# Charger le modèle CNN
model_cancer = load_model('model_cancer.keras')

class_mapping = {
    0: 'breast_benign',
    1: 'breast_malignant',
    2: 'breast_normal',
    3: 'lung_benign',
    4: 'lung_malignant',
    5: 'lung_normal',
    6: 'skin_benign',
    7: 'skin_malignant',
    8: 'tumer_brain_benign',
    9: 'tumer_brain_malignant'
}
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
# Load the main prediction model
modele_chemin = "random_forest_model.pkl"
loaded_model = joblib.load(modele_chemin)

symptoms_list = ['itching', 'skin rash', 'nodal skin eruptions',
                'continuous sneezing', 'shivering', 'chills', 'joint pain',
                'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
                'vomiting', 'burning micturition', 'spotting urination', 'fatigue',
                'weight gain', 'anxiety', 'cold hands and feets', 'mood swings',
                'weight loss', 'restlessness', 'lethargy', 'patches in throat',
                'irregular sugar level', 'cough', 'high fever', 'sunken eyes',
                'breathlessness', 'sweating', 'dehydration', 'indigestion',
                'headache', 'yellowish skin', 'dark urine', 'nausea',
                'loss of appetite', 'pain behind the eyes', 'back pain',
                'constipation', 'abdominal pain', 'diarrhoea', 'mild fever',
                'yellow urine', 'yellowing of eyes', 'acute liver failure',
                'fluid overload', 'swelling of stomach', 'swelled lymph nodes',
                'malaise', 'blurred and distorted vision', 'phlegm',
                'throat irritation', 'redness of eyes', 'sinus pressure',
                'runny nose', 'congestion', 'chest pain', 'weakness in limbs',
                'fast heart rate', 'pain during bowel movements',
                'pain in anal region', 'bloody stool', 'irritation in anus',
                'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity',
                'swollen legs', 'swollen blood vessels', 'puffy face and eyes',
                'enlarged thyroid', 'brittle nails', 'swollen extremeties',
                'excessive hunger', 'extra marital contacts',
                'drying and tingling lips', 'slurred speech', 'knee pain',
                'hip joint pain', 'muscle weakness', 'stiff neck',
                'swelling joints', 'movement stiffness', 'spinning movements',
                'loss of balance', 'unsteadiness', 'weakness of one body side',
                'loss of smell', 'bladder discomfort', 'foul smell ofurine',
                'continuous feel of urine', 'passage of gases', 'internal itching',
                'toxic look (typhos)', 'depression', 'irritability', 'muscle pain',
                'altered sensorium', 'red spots over body', 'belly pain',
                'abnormal menstruation', 'dischromic patches',
                'watering from eyes', 'increased appetite', 'polyuria',
                'family history', 'mucoid sputum', 'rusty sputum',
                'lack of concentration', 'visual disturbances',
                'receiving blood transfusion', 'receiving unsterile injections',
                'coma', 'stomach bleeding', 'distention of abdomen',
                'history of alcohol consumption', 'blood in sputum',
                'prominent veins on calf', 'palpitations', 'painful walking',
                'pus filled pimples', 'blackheads', 'scurring', 'skin peeling',
                'silver like dusting', 'small dents in nails',
                'inflammatory nails', 'blister', 'red sore around nose',
                'yellow crust ooze', 'prognosis']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

@app.route('/sante')
def sante():
    return render_template('sante.html')

@app.route('/corona', methods=['GET', 'POST'])
def corona():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        pouls = float(request.form['pouls'])
        oxygene = float(request.form['oxygene'])
        glycemie = float(request.form['glycemie'])
        tension = float(request.form['tension'])

        user_data = pd.DataFrame({
            'temperature': [temperature],
            'pouls': [pouls],
            'oxygene': [oxygene],
            'glycemie': [glycemie],
            'tension': [tension]
        })

        covid_prediction = model_covid.predict(user_data)

        return render_template('resultat.html', prediction=covid_prediction[0])

    return render_template('corona.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = int(request.form['age'])

        user_data_diabetes = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })

        prediction_diabetes1 = model_diabetes1.predict(user_data_diabetes)
        prediction_diabetes2 = model_diabetes2.predict(user_data_diabetes)
        prediction_diabetes3 = model_diabetes3.predict(user_data_diabetes)
        prediction_diabetes4 = model_diabetes4.predict(user_data_diabetes)

        return render_template('result_diabetes.html', 
                               prediction1=prediction_diabetes1[0],
                               prediction2=prediction_diabetes2[0],
                               prediction3=prediction_diabetes3[0],
                               prediction4=prediction_diabetes4[0])

    return render_template('diabetes.html')




@app.route('/cancer', methods=['GET', 'POST'])
def cancer():
    if request.method == 'POST':
        # Vérifier si le fichier est présent dans la requête
        if 'image_file' not in request.files:
            return 'Aucun fichier téléchargé'

        # Obtenir le fichier téléchargé
        file = request.files['image_file']

        # Vérifier si le fichier est vide
        if file.filename == '':
            return 'Fichier non sélectionné'

        # Générer un nom de fichier unique
        unique_filename = str(uuid.uuid4()) + '.jpg'

        # Chemin complet du fichier dans le répertoire de sauvegarde
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Sauvegarder le fichier dans un emplacement temporaire
        file.save(file_path)

        # Prédiction sur l'image
        preprocessed_image = preprocess_image(file_path)
        prediction = model_cancer.predict(preprocessed_image)
        predicted_classes = np.argmax(prediction, axis=1)
        predicted_disease = class_mapping[predicted_classes[0]]

        # Supprimer le fichier temporaire après utilisation
        os.remove(file_path)

        # Renvoyer le résultat à la page prediction.html
        return render_template('resultat cancer.html', image_path=file_path, predicted_disease=predicted_disease)

    return render_template('cancer.html')

# Charger les données nécessaires
df1 = pd.read_csv("C:\\Users\\lenovo\\Downloads\\apro\\static\\diabetes.csv")
discrp = pd.read_csv("C:\\Users\\lenovo\\Downloads\\apro\\static\\symptom_Description.csv")
ektra7at = pd.read_csv("C:\\Users\\lenovo\\Downloads\\apro\\static\\symptom_precaution.csv")



# Fonction de prédiction
def predd(model, *args):
    # Transformer les symptômes en poids
    weights = symptoms_to_weights(args)

    # Assurez-vous que weights contient exactement 17 éléments
    while len(weights) < 17:
        weights.append(0)
    
    print("Poids des symptômes :", weights)  # Afficher les poids des symptômes pour le débogage
    
    # Effectuez la prédiction avec le modèle
    pred2 = model.predict([weights])

    print("Prédiction du modèle :", pred2)  # Afficher la prédiction du modèle pour le débogage
    
    # Récupérer la description de la maladie prédite
    disp = discrp[discrp['Disease'] == pred2[0]].values[0][1]

    # Récupérer les recommandations pour la maladie prédite
    recomnd = ektra7at[ektra7at['Disease'] == pred2[0]]
    c = np.where(ektra7at['Disease'] == pred2[0])[0][0]
    precuation_list = list(ektra7at.iloc[c, 1:])

    # Afficher les résultats dans la console
    print("Nom de la maladie :", pred2[0])
    print("Description de la maladie :", disp)
    print("Précautions recommandées :")
    for precaution in precuation_list:
        print(precaution)

    # Retournez les résultats si nécessaire
    return pred2[0], disp, precuation_list



# Définir les poids des symptômes
symptoms_weights = {
    'itching': 1,
    'skin_rash': 3,
    'nodal_skin_eruptions': 4,
    'continuous_sneezing': 4,
    'shivering': 5,
    'chills': 3,
    'joint_pain': 3,
    'stomach_pain': 5,
    'acidity': 3,
    'ulcers_on_tongue': 4,
    'muscle_wasting': 3,
    'vomiting': 5,
    'burning_micturition': 6,
    'spotting_urination': 6,
    'fatigue': 4,
    'weight_gain': 3,
    'anxiety': 4,
    'cold_hands_and_feets': 5,
    'mood_swings': 3,
    'weight_loss': 3,
    'restlessness': 5,
    'lethargy': 2,
    'patches_in_throat': 6,
    'irregular_sugar_level': 5,
    'cough': 4,
    'high_fever': 7,
    'sunken_eyes': 3,
    'breathlessness': 4,
    'sweating': 3,
    'dehydration': 4,
    'indigestion': 5,
    'headache': 3,
    'yellowish_skin': 3,
    'dark_urine': 4,
    'nausea': 5,
    'loss_of_appetite': 4,
    'pain_behind_the_eyes': 4,
    'back_pain': 3,
    'constipation': 4,
    'abdominal_pain': 4,
    'diarrhoea': 6,
    'mild_fever': 5,
    'yellow_urine': 4,
    'yellowing_of_eyes': 4,
    'acute_liver_failure': 6,
    'fluid_overload': 6,
    'swelling_of_stomach': 7,
    'swelled_lymph_nodes': 6,
    'malaise': 6,
    'blurred_and_distorted_vision': 5,
    'phlegm': 5,
    'throat_irritation': 4,
    'redness_of_eyes': 5,
    'sinus_pressure': 4,
    'runny_nose': 5,
    'congestion': 5,
    'chest_pain': 7,
    'weakness_in_limbs': 7,
    'fast_heart_rate': 5,
    'pain_during_bowel_movements': 5,
    'pain_in_anal_region': 6,
    'bloody_stool': 5,
    'irritation_in_anus': 6,
    'neck_pain': 5,
    'dizziness': 4,
    'cramps': 4,
    'bruising': 4,
    'obesity': 4,
    'swollen_legs': 5,
    'swollen_blood_vessels': 5,
    'puffy_face_and_eyes': 5,
    'enlarged_thyroid': 6,
    'brittle_nails': 5,
    'swollen_extremeties': 5,
    'excessive_hunger': 4,
    'extra_marital_contacts': 5,
    'drying_and_tingling_lips': 4,
    'slurred_speech': 4,
    'knee_pain': 3,
    'hip_joint_pain': 2,
    'muscle_weakness': 2,
    'stiff_neck': 4,
    'swelling_joints': 5,
    'movement_stiffness': 5,
    'spinning_movements': 6,
    'loss_of_balance': 4,
    'unsteadiness': 4,
    'weakness_of_one_body_side': 4,
    'loss_of_smell': 3,
    'bladder_discomfort': 4,
    'foul_smell_ofurine': 5,
    'continuous_feel_of_urine': 6,
    'passage_of_gases': 5,
    'internal_itching': 4,
    'toxic_look_(typhos)': 5,
    'depression': 3,
    'irritability': 2,
    'muscle_pain': 2,
    'altered_sensorium': 2,
    'red_spots_over_body': 3,
    'belly_pain': 4,
    'abnormal_menstruation': 6,
    'dischromic_patches': 6,
    'watering_from_eyes': 4,
    'increased_appetite': 5,
    'polyuria': 4,
    'family_history': 5,
    'mucoid_sputum': 4,
    'rusty_sputum': 4,
    'lack_of_concentration': 3,
    'visual_disturbances': 3,
    'receiving_blood_transfusion': 5,
    'receiving_unsterile_injections': 2,
    'coma': 7,
    'stomach_bleeding': 6,
    'distention_of_abdomen': 4,
    'history_of_alcohol_consumption': 5,
    'fluid_overload': 4,
    'blood_in_sputum': 5,
    'prominent_veins_on_calf': 6,
    'palpitations': 4,
    'painful_walking': 2,
    'pus_filled_pimples': 2,
    'blackheads': 2,
    'scurring': 2,
    'skin_peeling': 3,
    'silver_like_dusting': 2,
    'small_dents_in_nails': 2,
    'inflammatory_nails': 2,
    'blister': 4,
    'red_sore_around_nose': 2,
    'yellow_crust_ooze': 3,
    'prognosis': 5
}

# Fonction pour transformer les symptômes en poids
def symptoms_to_weights(symptoms_list):
    weights = [symptoms_weights[symptom] if symptom in symptoms_weights else 0 for symptom in symptoms_list]
    return weights



@app.route('/Prédiction Générale', methods=['GET', 'POST'])
def prediction_generale_route():
    if request.method == 'POST':
        # Obtenir les symptômes sélectionnés dans le formulaire
        selected_symptoms = request.form.getlist('symptoms[]')

        # Vérifier si des symptômes ont été sélectionnés
        if not selected_symptoms:
            # Aucun symptôme sélectionné, renvoyer un message d'erreur
            return render_template('error.html', message="Aucun symptôme sélectionné")

        # Appeler la fonction predd avec les symptômes sélectionnés
        prediction = predd(loaded_model, *selected_symptoms)

        # Renvoyer les résultats au modèle de résultat
        return render_template('resultat_generale.html', prediction=prediction)

    # Rendre le formulaire HTML pour sélectionner les symptômes
    return render_template('Prédiction Générale.html', symptoms=symptoms_list, symptoms_count=len(symptoms_list))


if __name__ == '__main__':
    app.run(debug=True)
