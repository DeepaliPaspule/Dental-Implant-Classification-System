# Dental Implant Classification System 🦷

A deep learning system for classifying dental implants using PyTorch, Flask, and Streamlit. The system identifies four types of dental implants: Endosteal, Subperiosteal, Transosteal, and Zygomatic.

## 📋 Features
- **Real-time Classification**: Upload and classify dental implant images instantly
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Performance Metrics**: Confusion matrix, ROC curves, and class-wise metrics
- **REST API**: Flask backend for easy integration

## 🚀 Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/dental_implant_prediction.git
cd dental_implant_prediction 


2. Create and activate virtual environment:
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. Install dependencies:
bash
pip install -r requirements.txt


## 💻 Usage

1. Start the Flask API:
bash
python src/app.py --port 5001


2. Run the Streamlit frontend:
bash
streamlit run src/frontend.py

3. Access the web interface at `http://localhost:8501`


## 📊 Performance Metrics
- Accuracy: 34.42%
- Precision: 13.81%
- Recall: 34.42%

Per-class Performance:
| Class         | Precision | Recall | F1 Score |
|--------------|-----------|---------|----------|
| Endosteal    | 0.XX     | 0.XX   | 0.XX    |
| Subperiosteal| 0.XX     | 0.XX   | 0.XX    |
| Transosteal  | 0.XX     | 0.XX   | 0.XX    |
| Zygomatic    | 0.XX     | 0.XX   | 0.XX    |


## 📚 Dataset
Total Images: 5,107
- Endosteal: 1,970 images
- Subperiosteal: 511 images
- Transosteal: 704 images
- Zygomatic: 1,922 images

## 🔌 API Usage
python
import requests
url = 'http://localhost:5001/predict'
files = {'image': open('path/to/image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())


## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author
- Deepali Ravindra Paspule - [GitHub](https://github.com/DeepaliPaspule)

---
Made with ❤️ by Deepali
