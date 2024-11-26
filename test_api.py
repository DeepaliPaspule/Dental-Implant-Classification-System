import requests
import os

def test_prediction():
    # URL of your Flask app
    url = 'http://127.0.0.1:5001/predict'
    
    # Get the absolute path to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to a test image - using one from your dataset
    test_image_path = os.path.join(
        project_dir,
        'data', 'raw', 'EntireDataset', 'zygomatic', 
        'images', '13', '13_1.jpg'
    )
    
    print(f"Looking for test image at: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        # Try another image from your dataset
        test_image_path = os.path.join(
            project_dir,
            'data', 'raw', 'EntireDataset', 'transosteal', 
            'images', '9', '9_1.jpg'
        )
        print(f"Trying alternative image at: {test_image_path}")
        if not os.path.exists(test_image_path):
            print(f"Error: Alternative test image not found either")
            return
    
    try:
        # Prepare the image file
        files = {
            'image': ('test_image.jpg', open(test_image_path, 'rb'), 'image/jpeg')
        }
        
        print(f"Sending request to {url}...")
        
        # Make the POST request
        response = requests.post(url, files=files)
        
        # Print the response
        print('Status Code:', response.status_code)
        print('Response:', response.json())
        
    except Exception as e:
        print(f"Error during API test: {str(e)}")
    
if __name__ == '__main__':
    test_prediction()
