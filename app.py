from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Load model from disk
model_path = os.path.join(os.getcwd(), 'classifier_rf.pkl')

def load_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None

model = load_model()

if model is None:
    print("Model not found! Please ensure the 'classifier_rf.pkl' file exists.")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Retrieve form data
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Validate input data
            if None in [N, P, K, temperature, humidity, ph, rainfall]:
                flash("Please fill in all fields correctly.", "error")
                return redirect(url_for('home'))
            
            # Create DataFrame for prediction
            input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                      columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

            # Make prediction
            prediction = model.predict(input_data)[0]
            
            return render_template('index.html', prediction=prediction)
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for('home'))

    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
