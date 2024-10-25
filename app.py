import pandas as pd
from flask import Flask, render_template, request
import joblib

# Load the pre-trained model and encoders
model = joblib.load('flight_model.joblib')
airline_encode = joblib.load('airline_encode')
flight_encode = joblib.load('flight_encode')
source_city_encode = joblib.load('source_city_encode')
departure_time_encode = joblib.load('departure_time_encode')
stops_encode = joblib.load('stops_encode')
arrival_time_encode = joblib.load('arrival_time_encode')
destination_city_encode = joblib.load('destination_city_encode')
class_encode = joblib.load('class_encode')

# Initialize the Flask app
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html template

# Define the route for handling form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user input from the form and convert to appropriate types
        user_data = {
            'airline': request.form['airline'],
            'flight': request.form['flight'],
            'source_city': request.form['source_city'],
            'departure_time': request.form['departure_time'],
            'stops': request.form['stops'],
            'arrival_time': request.form['arrival_time'],
            'destination_city': request.form['destination_city'],
            'class': request.form['class'],
            'duration': float(request.form['duration']),
            'days_left': int(request.form['days_left'])
        }

        # Encode the user input using the loaded encoders
        airline_encoded = airline_encode.transform([user_data['airline']])[0]
        flight_encoded = flight_encode.transform([user_data['flight']])[0]
        source_city_encoded = source_city_encode.transform([user_data['source_city']])[0]
        departure_time_encoded = departure_time_encode.transform([user_data['departure_time']])[0]
        stops_encoded = stops_encode.transform([user_data['stops']])[0]
        arrival_time_encoded = arrival_time_encode.transform([user_data['arrival_time']])[0]
        destination_city_encoded = destination_city_encode.transform([user_data['destination_city']])[0]
        class_encoded = class_encode.transform([user_data['class']])[0]

        # Prepare the input data for prediction
        data = pd.DataFrame([[airline_encoded, flight_encoded, source_city_encoded, departure_time_encoded, stops_encoded, arrival_time_encoded, destination_city_encoded, class_encoded, user_data['duration'], user_data['days_left']]],
                            columns=['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left'])

        # Make prediction using the loaded model
        prediction = model.predict(data)[0]

        # Render the result template with the prediction and user data
        return render_template('result.html', prediction=prediction, user_data=user_data)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode for development