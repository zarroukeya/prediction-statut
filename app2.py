import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Connexion à la base de données
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-RSFN8HH\SQLEXPRESS;DATABASE=immobilier;UID=sa;PWD=12356')
cursor = conn.cursor()    # Execute SQL query
query = "SELECT * FROM Dimm_client"
df = pd.read_sql(query, conn)

# Sélection des colonnes appropriées pour les caractéristiques et la variable cible
X = df[['activite', 'region', 'Status_Client','methode_paiement','montant']]  
y = df['Statut_paiement']  

# Convertir les colonnes catégorielles en type de données category
X_categorical = X.astype('category')

# Convertir les colonnes catégorielles en valeurs numériques
X_encoded = X_categorical.apply(lambda x: x.cat.codes)

# Convertir les étiquettes de la variable cible en valeurs numériques
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Entraîner le modèle XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Définition de l'API FastAPI
app = FastAPI()

# Middleware CORS pour autoriser les requêtes de n'importe quel domaine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Définition du schéma de la requête
class InputData(BaseModel):
    activite: str
    region: str
    Status_Client: str
    methode_paiement: str
    montant: float

# Définition de la route pour la prédiction
@app.post("/predict_ps/")
async def predict_status(data: InputData):
    # Créer un DataFrame à partir des données d'entrée
    input_data = pd.DataFrame([data.dict()])
    
    # Convertir les colonnes catégorielles en valeurs numériques
    input_data_categorical = input_data.astype('category')
    input_data_encoded = input_data_categorical.apply(lambda x: x.cat.codes)
    
    # Effectuer la prédiction
    prediction = xgb.predict(input_data_encoded)
    
    # Convertir la prédiction en libellé de statut de paiement
    predicted_status = label_encoder_y.inverse_transform(prediction)[0]
    
    return {"predicted_status": predicted_status}
