# 🥔 Potato Disease Prediction using Deep Learning

A deep learning-based image classification system that identifies three types of potato leaf conditions: **Early Blight**, **Late Blight**, and **Healthy**. This project leverages Convolutional Neural Networks (CNN) to help detect potato diseases early, potentially assisting farmers in taking timely action to protect their crops.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://potato-disease-prediction-model.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

## 🌟 Live Application

Try the model in action: **[Potato Disease Prediction App](https://potato-disease-prediction-model.streamlit.app/)**

Simply upload an image of a potato leaf and get instant predictions!

## 📋 Table of Contents

- [Overview](#overview)
- [Disease Classes](#disease-classes)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Potato diseases like Early Blight and Late Blight can cause significant crop losses, affecting both yield and quality. Early detection is crucial for effective treatment and disease management. This project uses deep learning to classify potato leaf images into three categories, providing a quick and accessible tool for disease identification.

## 🍃 Disease Classes

The model can identify the following three conditions:

1. **Early Blight** (*Alternaria solani*)
   - Causes dark spots with concentric rings on leaves
   - Reduces yield and quality
   - Requires specific fungicide treatment

2. **Late Blight** (*Phytophthora infestans*)
   - Historical significance: Caused the Irish Potato Famine
   - Appears as water-soaked lesions on leaves
   - Can destroy entire crops rapidly
   - Requires immediate intervention

3. **Healthy**
   - No visible disease symptoms
   - Normal, green, and thriving leaves

## 📊 Dataset

**Source**: [PlantVillage Potato Disease Dataset on Kaggle](https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset)

### Dataset Details:
- **Total Images**: 2,152 images
- **Classes**: 3 (Early Blight, Late Blight, Healthy)
- **Image Format**: JPG/PNG
- **Resolution**: Resized to 256x256 pixels for training
- **Source**: PlantVillage Project - a publicly available plant disease dataset

### Download Dataset:
```python
import kagglehub

# Download the dataset
path = kagglehub.dataset_download("aarishasifkhan/plantvillage-potato-disease-dataset")
print("Dataset path:", path)
```

Alternatively, download directly from Kaggle: [PlantVillage Potato Disease Dataset](https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset)

## 🏗️ Model Architecture

### Convolutional Neural Network (CNN)
The model is built using TensorFlow/Keras with the following architecture:

- **Input Layer**: 256x256x3 (RGB images)
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Data Augmentation**: 
  - Random flipping
  - Random rotation
  - Random zoom
- **Regularization**: Dropout layers to prevent overfitting
- **Output Layer**: Dense layer with softmax activation (3 classes)

### Training Configuration:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Image Size**: 256x256 pixels
- **GPU Acceleration**: Trained on NVIDIA Tesla T4 (Kaggle)

## ✨ Features

- ✅ **Real-time Prediction**: Upload images and get instant disease classification
- ✅ **High Accuracy**: Achieves ~90-95% accuracy on test data
- ✅ **User-Friendly Interface**: Built with Streamlit for easy accessibility
- ✅ **Three-Class Classification**: Early Blight, Late Blight, and Healthy
- ✅ **Responsive Design**: Works on desktop and mobile devices
- ✅ **Fast Inference**: Optimized model for quick predictions

## 🛠️ Technology Stack

### Core Technologies:
- **Python 3.8+**: Primary programming language
- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural networks API
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization

### Deployment:
- **Streamlit**: Web application framework
- **Streamlit Cloud**: Hosting platform

### Development Environment:
- **Kaggle Notebooks**: Model training with GPU support
- **Jupyter Notebook**: Experimentation and analysis

## 🚀 Installation

### Prerequisites:
- Python 3.8 or higher
- pip package manager

### Clone the Repository:
```bash
git clone https://github.com/Suraj8Sharma/POTATO-DISEASE-PREDICTION.git
cd POTATO-DISEASE-PREDICTION
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages:
```
tensorflow>=2.8.0
streamlit>=1.10.0
numpy>=1.21.0
pillow>=9.0.0
matplotlib>=3.5.0
kagglehub
```

## 💻 Usage

### Running the Streamlit App Locally:
```bash
streamlit run frontend/app.py
```

The app will open in your default browser at `http://localhost:8501`

### Making Predictions:
1. Open the web application
2. Upload an image of a potato leaf (JPG or PNG)
3. Click "Predict"
4. View the classification result and confidence score

### Using the Jupyter Notebook:
```bash
jupyter notebook potato-disease-prediciton.ipynb
```

## 📈 Model Performance

- **Training Accuracy**: ~92-94%
- **Validation Accuracy**: ~90-95%
- **Test Performance**: Robust across different leaf conditions
- **Inference Time**: <1 second per image

### Sample Predictions:
The model successfully identifies:
- Early Blight lesions with characteristic concentric rings
- Late Blight water-soaked patches
- Healthy green leaves without disease symptoms

## 📁 Project Structure

```
POTATO-DISEASE-PREDICTION/
│
├── frontend/                      # Streamlit web application
│   └── app.py                     # Main application file
│
├── potato-disease-prediciton.ipynb # Training notebook
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

## 🔮 Future Enhancements

- [ ] Add more disease classes (e.g., bacterial diseases, nutrient deficiencies)
- [ ] Implement mobile application (Android/iOS)
- [ ] Add disease treatment recommendations
- [ ] Include severity assessment for detected diseases
- [ ] Multi-language support for broader accessibility
- [ ] Integration with agricultural IoT devices
- [ ] Real-time batch processing for multiple images
- [ ] Model optimization for edge devices (Raspberry Pi, mobile)

## 🌾 Real-World Applications

- **Precision Agriculture**: Early disease detection for targeted treatment
- **Crop Management**: Monitor large fields efficiently
- **Educational Tool**: Help students and farmers learn about potato diseases
- **Research**: Support agricultural research and disease pattern analysis
- **Cost Reduction**: Minimize crop losses through early intervention

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution:
- Improving model accuracy
- Adding new features to the web app
- Expanding dataset with more images
- Optimizing inference speed
- Documentation improvements

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Suraj Sharma**

- GitHub: [@Suraj8Sharma](https://github.com/Suraj8Sharma)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/suraj8sharma)

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing the comprehensive potato disease image dataset
- **Kaggle**: For providing free GPU resources for model training
- **TensorFlow Team**: For the excellent deep learning framework
- **Streamlit**: For the intuitive web app framework
- **Agricultural Community**: For inspiration and real-world use cases

## 📚 References

- [PlantVillage Dataset Paper](https://arxiv.org/abs/1511.08060)
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Deep Learning for Plant Disease Detection](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)

## 📧 Contact

For questions, suggestions, or collaboration opportunities, feel free to reach out:

- Create an issue in this repository
- Connect on LinkedIn
- Email: [Your email if you want to add]

---

⭐ If you find this project helpful, please consider giving it a star!

**Live Demo**: [Try it now!](https://potato-disease-prediction-model.streamlit.app/)
