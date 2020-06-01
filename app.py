# Exposing the model via an API
import joblib
from flask import Flask, request
import numpy as np
import pandas as pd
from flask_swagger_ui import get_swaggerui_blueprint


# Load the model
model = joblib.load('models/clf.pkl')


# Initiate the Flask App
app = Flask(__name__)

# swagger specific
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGER_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app name': "Machine learning Model Engines"
    }
)
app.register_blueprint(SWAGGER_BLUEPRINT, url_prefix=SWAGGER_URL)


@app.route('/predict')
def predicti_iris():

    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")

    # Convrt to numpy array and ensure the other of the input features
    prediction = model.predict(
        np.array([[s_length, s_width, p_length, p_width]]))

    return str(prediction)


@app.route('/predict_file', methods=['POST'])
def predict_file():

    input_data = pd.read_csv(request.files.get('input_file'), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=3000)
    app.run(debug=True, port=3000)
