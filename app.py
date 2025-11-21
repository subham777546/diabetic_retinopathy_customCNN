from flask import Flask, request, jsonify
from flask_cors import CORS
import model_inference_backend
import os
app = Flask(__name__)
CORS(app) 

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handles POST requests with a fundus image and returns the DR diagnosis.
    """

    if 'fundus_image' not in request.files:
        return jsonify({"error": "No image file provided in request."}), 400

    image_file = request.files['fundus_image']

    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Unsupported file type. Please use PNG or JPG."}), 400

    try:

        image_bytes = image_file.read()

        result = model_inference_backend.predict_image(image_bytes)

        if result.get("error"):

             return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        print(f"Unhandled error during prediction: {e}")
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

if __name__ == '__main__':

    if model_inference_backend.MODEL is not None:
        print("Starting Flask server on http://127.0.0.1:5000")

        app.run(debug=True, port=5000, use_reloader=False)
    else:
        print("Server NOT started due to model loading failure. Check model_inference_backend.py logs.")
