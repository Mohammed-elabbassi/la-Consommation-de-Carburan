from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle et le scaler
model, scaler = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des valeurs du formulaire
        features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform([features])

        prediction = model.predict(scaled_features)[0]

        return render_template(
            'result.html',
            prediction_text=f"Consommation estimée : {prediction:.2f} miles/gallon"
        )
    except Exception as e:
        return render_template('result.html', prediction_text=f"Erreur : {e}")

if __name__ == '__main__':
    app.run(debug=True)
