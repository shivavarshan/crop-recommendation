import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('classifier_rf.pkl', 'rb'))

# Create the Streamlit app
st.title('Crop Recommendation App')

# Get input values from the user with placeholders
N = st.number_input('Nitrogen (N)', placeholder='Enter N value (e.g., 90)', value=None)
P = st.number_input('Phosphorus (P)', placeholder='Enter P value (e.g., 42)', value=None)
K = st.number_input('Potassium (K)', placeholder='Enter K value (e.g., 43)', value=None)
temperature = st.number_input('Temperature (°C)', placeholder='Enter temperature in Celsius (e.g., 20.88)', value=None)
humidity = st.number_input('Humidity (%)', placeholder='Enter humidity percentage (e.g., 82.00)', value=None)
ph = st.number_input('pH', placeholder='Enter pH value (e.g., 6.50)', value=None)
rainfall = st.number_input('Rainfall (mm)', placeholder='Enter rainfall in millimeters (e.g., 202.94)', value=None)

# Check if all inputs are provided
if st.button('Submit'):
    if None in [N, P, K, temperature, humidity, ph, rainfall]:
        st.warning("Please fill in all the input fields!")
    else:
        # Create a dataframe from the input values
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Display the prediction with custom styling
        st.markdown(f'<p class="result">The recommended crop for the given conditions is: {prediction}</p>', unsafe_allow_html=True)
