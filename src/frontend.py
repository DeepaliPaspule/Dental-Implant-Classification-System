import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from predict import DentalImplantPredictor
from config import Config
import torch

class DentalImplantFrontend:
    def __init__(self):
        self.predictor = DentalImplantPredictor()
        self.setup_page()
        
    def setup_page(self):
        st.set_page_config(
            page_title="Dental Implant Classifier",
            page_icon="🦷",
            layout="wide"
        )
        st.title("🦷 Dental Implant Classification System")
        
    def run(self):
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Choose a page",
            ["Home", "Predict", "Model Performance", "Dataset Analysis"]
        )
        
        if page == "Home":
            self.show_home()
        elif page == "Predict":
            self.show_prediction_page()
        elif page == "Model Performance":
            self.show_performance_page()
        elif page == "Dataset Analysis":
            self.show_dataset_analysis()

    def show_home(self):
        st.header("Welcome to the Dental Implant Classification System")
        
        st.markdown("""
        This application helps classify different types of dental implants:
        - Endosteal
        - Subperiosteal
        - Transosteal
        - Zygomatic
        
        ### Features
        - Real-time implant classification
        - Detailed model performance metrics
        - Dataset analysis and visualization
        - Interactive confusion matrix
        
        ### How to Use
        1. Navigate to the 'Predict' page to classify new implant images
        2. Check 'Model Performance' to see detailed metrics
        3. Explore 'Dataset Analysis' for data distribution insights
        """)

    def show_prediction_page(self):
        st.header("Implant Classification")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with col2:
                if st.button('Classify'):
                    with st.spinner('Processing...'):
                        # Make prediction
                        result = self.predictor.predict(uploaded_file)
                        
                        # Display results
                        st.success('Classification Complete!')
                        st.write(f"**Predicted Class:** {result['class']}")
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        
                        # Create confidence chart
                        fig = go.Figure(go.Bar(
                            x=[result['confidence']],
                            y=[result['class']],
                            orientation='h'
                        ))
                        fig.update_layout(
                            title="Prediction Confidence",
                            xaxis_title="Confidence",
                            yaxis_title="Class"
                        )
                        st.plotly_chart(fig)

    def show_performance_page(self):
        st.header("Model Performance Metrics")
        
        # Load or compute metrics
        metrics = self.load_metrics()
        
        # Display overall metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.metric_card("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            self.metric_card("Precision", f"{metrics['precision']:.2%}")
        with col3:
            self.metric_card("Recall", f"{metrics['recall']:.2%}")
        with col4:
            self.metric_card("F1 Score", f"{metrics['f1']:.2%}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig_cm = self.plot_confusion_matrix(metrics['confusion_matrix'])
        st.pyplot(fig_cm)
        
        # ROC Curve
        st.subheader("ROC Curves")
        fig_roc = self.plot_roc_curves(metrics['roc_curves'])
        st.plotly_chart(fig_roc)
        
        # Per-class metrics
        st.subheader("Per-class Performance")
        self.show_class_metrics(metrics['class_metrics'])

    def show_dataset_analysis(self):
        st.header("Dataset Analysis")
        
        # Class distribution
        st.subheader("Class Distribution")
        class_dist = self.get_class_distribution()
        fig = px.bar(
            class_dist, 
            x='class', 
            y='count',
            title="Number of Images per Class"
        )
        st.plotly_chart(fig)
        
        # Image size distribution
        st.subheader("Image Size Distribution")
        size_dist = self.get_image_size_distribution()
        fig = px.scatter(
            size_dist,
            x='width',
            y='height',
            color='class',
            title="Image Dimensions by Class"
        )
        st.plotly_chart(fig)

    @staticmethod
    def metric_card(title, value):
        st.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6;">
                <h3 style="margin: 0; font-size: 1rem;">{title}</h3>
                <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    def load_metrics(self):
        # Placeholder - replace with actual metrics computation
        return {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.84,
            'f1': 0.835,
            'confusion_matrix': np.random.rand(4, 4),
            'roc_curves': {
                'fpr': np.linspace(0, 1, 100),
                'tpr': np.linspace(0, 1, 100)
            },
            'class_metrics': {
                class_name: {
                    'precision': np.random.random(),
                    'recall': np.random.random(),
                    'f1': np.random.random()
                }
                for class_name in Config.CATEGORIES
            }
        }

    def plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            xticklabels=Config.CATEGORIES,
            yticklabels=Config.CATEGORIES,
            ax=ax
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return fig

    def plot_roc_curves(self, roc_data):
        fig = go.Figure()
        for class_name in Config.CATEGORIES:
            fig.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                name=f'ROC {class_name}'
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(dash='dash'),
            name='Random'
        ))
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        return fig

    def show_class_metrics(self, metrics):
        df = pd.DataFrame(metrics).T
        st.dataframe(df)

    def get_class_distribution(self):
        # Placeholder - replace with actual data
        return pd.DataFrame({
            'class': Config.CATEGORIES,
            'count': np.random.randint(100, 1000, len(Config.CATEGORIES))
        })

    def get_image_size_distribution(self):
        # Placeholder - replace with actual data
        data = []
        for class_name in Config.CATEGORIES:
            n_samples = 100
            data.extend([{
                'width': np.random.randint(200, 800),
                'height': np.random.randint(200, 800),
                'class': class_name
            } for _ in range(n_samples)])
        return pd.DataFrame(data)

if __name__ == "__main__":
    frontend = DentalImplantFrontend()
    frontend.run()