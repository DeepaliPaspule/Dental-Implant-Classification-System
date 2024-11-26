from flask import Flask, request, jsonify
import os
from predict import DentalImplantPredictor
from config import Config

app = Flask(__name__)
predictor = None

def init_predictor():
    global predictor
    if predictor is None:
        print("Initializing predictor...")
        predictor = DentalImplantPredictor()
        print("Predictor initialized successfully")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    init_predictor()
    
    if 'image' not in request.files:
        print("No image file in request")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the file temporarily
        os.makedirs('temp', exist_ok=True)
        temp_path = os.path.join('temp', file.filename)
        file.save(temp_path)
        
        print(f"Processing image: {temp_path}")
        result = predictor.predict_image(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        print(f"Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    Config.print_paths()
    app.run(host='0.0.0.0', port=5001, debug=True)