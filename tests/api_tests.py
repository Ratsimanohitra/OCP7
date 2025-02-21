import pytest
import pandas as pd
import numpy as np
from flask import Flask, json
from source.api import app  # Replace with the name of your Flask app file

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def random_row():
    """Load a random row from the Parquet file."""
    df = pd.read_parquet('./data/data.parquet')  # Replace with the path to your Parquet file
    return df.sample(n=1).drop('SK_ID_CURR', axis=1).values.tolist()[0]

def test_predict_endpoint(client, random_row):
    """Test the /predict endpoint with a random row from the Parquet file."""
    # Prepare the request payload
    payload = {'data': [random_row]}

    # Send a POST request to the /predict endpoint
    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )

    # Check the response status code
    assert response.status_code == 200

    # Parse the response JSON
    response_data = json.loads(response.data)

    # Check if the response contains the 'prediction' key
    assert 'prediction' in response_data

    # Check if the prediction is a list
    assert isinstance(response_data['prediction'], list)

    # Check if the prediction list is not empty
    assert len(response_data['prediction']) > 0

    # Check if the prediction values are floats (probabilities)
    for pred in response_data['prediction']:
        assert isinstance(pred, float)
        assert 0 <= pred <= 1  # Ensure probabilities are between 0 and 1