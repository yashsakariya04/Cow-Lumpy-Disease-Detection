"""
Inference script for cow disease detection.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from typing import Union, List, Tuple, Optional
import argparse
import yaml
from PIL import Image
import streamlit as st


class CowDiseasePredictor:
    """Class for making predictions on cow images."""
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Healthy Cow', 'Lumpy Cow']
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.image_size = self.config['data']['image_size']
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Preprocessed image array
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image: Union[str, np.ndarray]) -> Tuple[str, float, np.ndarray]:
        """
        Make prediction on a single image.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please initialize the predictor properly.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        probabilities = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, probabilities[0]
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Tuple[str, float, np.ndarray]]:
        """
        Make predictions on multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            
        Returns:
            List of prediction tuples
        """
        predictions = []
        for image in images:
            try:
                pred = self.predict(image)
                predictions.append(pred)
            except Exception as e:
                print(f"Error processing image: {e}")
                predictions.append(("Error", 0.0, np.array([0.0, 0.0])))
        
        return predictions
    
    def visualize_prediction(self, image: Union[str, np.ndarray], 
                           save_path: Optional[str] = None, 
                           figsize: Tuple[int, int] = (12, 5)):
        """
        Visualize prediction with image and confidence scores.
        
        Args:
            image: Image path or numpy array
            save_path: Path to save the visualization
            figsize: Figure size
        """
        # Make prediction
        predicted_class, confidence, probabilities = self.predict(image)
        
        # Load original image for display
        if isinstance(image, str):
            display_img = cv2.imread(image)
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        else:
            display_img = image.copy()
            if len(display_img.shape) == 3 and display_img.shape[2] == 3:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Show image
        ax1.imshow(display_img)
        ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}')
        ax1.axis('off')
        
        # Show confidence scores
        bars = ax2.bar(self.class_names, probabilities, color=['green', 'red'])
        ax2.set_title('Prediction Probabilities')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add confidence values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "number_of_parameters": self.model.count_params(),
            "class_names": self.class_names
        }


def main():
    """Main function for command-line inference."""
    parser = argparse.ArgumentParser(description='Cow Disease Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to image for prediction')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--save', type=str, help='Path to save visualization')
    parser.add_argument('--batch', action='store_true', help='Process multiple images')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CowDiseasePredictor(args.model, args.config)
    
    if args.batch:
        # Process multiple images
        if os.path.isdir(args.image):
            image_files = [os.path.join(args.image, f) for f in os.listdir(args.image) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            # Assume it's a file with image paths
            with open(args.image, 'r') as f:
                image_files = [line.strip() for line in f.readlines()]
        
        predictions = predictor.predict_batch(image_files)
        
        for i, (img_path, pred) in enumerate(zip(image_files, predictions)):
            print(f"\nImage {i+1}: {img_path}")
            print(f"Prediction: {pred[0]}")
            print(f"Confidence: {pred[1]:.2%}")
    
    else:
        # Process single image
        try:
            predicted_class, confidence, probabilities = predictor.predict(args.image)
            
            print(f"\nImage: {args.image}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print(f"All Probabilities: {dict(zip(predictor.class_names, probabilities))}")
            
            # Visualize if save path provided
            if args.save:
                predictor.visualize_prediction(args.image, args.save)
            
        except Exception as e:
            print(f"Error: {e}")


# Streamlit app for web interface
def create_streamlit_app():
    """Create a Streamlit web interface for the predictor."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not available. Web interface disabled.")
        return
    
    st.set_page_config(
        page_title="Cow Disease Detection",
        page_icon="🐄",
        layout="wide"
    )
    
    st.title("🐄 Cow Lumpy Disease Detection")
    st.markdown("Upload an image of a cow to detect if it has lumpy skin disease.")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.text_input("Model Path", value="models/cow_disease_model_final.h5")
    
    # Initialize predictor
    if st.sidebar.button("Load Model"):
        try:
            predictor = CowDiseasePredictor(model_path)
            st.sidebar.success("Model loaded successfully!")
            st.session_state.predictor = predictor
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    
    # Main interface
    if 'predictor' in st.session_state:
        predictor = st.session_state.predictor
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image of a cow",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            if st.button("Analyze Image"):
                try:
                    # Convert PIL image to numpy array
                    img_array = np.array(image)
                    
                    # Make prediction
                    predicted_class, confidence, probabilities = predictor.predict(img_array)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction Results")
                        st.write(f"**Predicted Class:** {predicted_class}")
                        st.write(f"**Confidence:** {confidence:.2%}")
                        
                        # Confidence bar
                        if predicted_class == "Healthy Cow":
                            st.progress(confidence, text="Healthy Cow Confidence")
                        else:
                            st.progress(confidence, text="Lumpy Disease Confidence")
                    
                    with col2:
                        st.subheader("Detailed Probabilities")
                        for class_name, prob in zip(predictor.class_names, probabilities):
                            st.write(f"{class_name}: {prob:.2%}")
                    
                    # Model info
                    with st.expander("Model Information"):
                        model_info = predictor.get_model_info()
                        st.json(model_info)
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    
    else:
        st.info("Please load a model first using the sidebar.")


if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit
        if hasattr(streamlit, '_is_running_with_streamlit'):
            create_streamlit_app()
        else:
            main()
    except ImportError:
        main()
