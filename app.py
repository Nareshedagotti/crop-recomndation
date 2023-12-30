from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the Naive Bayes model and label encoder
naive_bayes_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('template.html')

# Endpoint to receive input data and make predictions using Naive Bayes
@app.route('/predict_nb', methods=['POST'])
def predict_naive_bayes():
    try:
        # Get input data from the form
        input_data = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall']),
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions using the Naive Bayes model
        nb_prediction = naive_bayes_model.predict(input_df)

        # Convert the prediction to a label
        label = label_encoder.inverse_transform(nb_prediction)[0]

        # Return the prediction as JSON
        return jsonify({'prediction': label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

