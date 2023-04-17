## NeuroAI

### Brain Tumor Detection Web App

This project involves the design and development of a web application for the detection of brain tumors in MRI images using deep learning techniques. The user-friendly interface provides options to either upload their own MRI image or select from a sample set of MRI images on the landing page. The prediction result, indicating the presence of a brain tumor, is displayed on the website with a clear and intuitive interface.

The application utilizes transfer learning with a pre-trained VGG model to extract features from MRI images, and a custom convolutional neural network (CNN) trained on a large dataset of MRI images to predict the presence of a brain tumor. These deep learning techniques, along with computer vision algorithms, are utilized to improve the accuracy of the tumor detection.

### Key Features
- Web application designed and developed using FastAPI framework
- Transfer learning with pre-trained VGG model to extract features from MRI images
- Custom convolutional neural network (CNN) trained on a large dataset of MRI images for improved accuracy of tumor prediction
- User-friendly interface with options to either upload their own MRI image or select from a sample set of MRI images
- Clear and intuitive display of prediction result on the website
- Utilization of deep learning techniques and computer vision algorithms to improve the accuracy of the tumor detection

### Requirements
- FastAPI
- TensorFlow
- Keras
- Numpy
- OpenCV

### Setup
1. Create a virtual env using 'python -m venv venv'
2. Clone this repository
3. Install the required packages using 'pip install -r requirements.txt'
4. Run the application using 'uvicorn app:app --reload'
5. Access the web app at 'http://localhost:8000' in your web browser.

### Output
1. Homepage:
![output-base](https://user-images.githubusercontent.com/37416550/232528459-9ea6288b-95cf-4e93-9dfc-6af6443ac08f.png)

2. After user chooses/uploads the brain MRI, the result of classification is displayed:
![output-prediction](https://user-images.githubusercontent.com/37416550/232529239-b86dd798-1b18-4eb3-8986-6762993f853c.png)



