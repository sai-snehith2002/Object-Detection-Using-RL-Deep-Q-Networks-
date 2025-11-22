# **Object Detection with RL Approach Using Deep Q-Networks**

This project presents a reinforcement learning (RL) driven object detection system where an agent learns to draw accurate bounding boxes around objects of interest. Instead of predicting bounding boxes directly like traditional detectors (YOLO, Faster R-CNN), the agent interacts with the image environment step-by-step. Through iterative actions and reward feedback, the bounding box is gradually refined until it correctly captures the object with high confidence.

The dataset consists of nine healthcare waste categories:

- Vials  
- Syringes  
- Glucose Drip  
- Plastic Bags  
- Cotton with Blood  
- IV Bags  
- Metal Items  
- IV Accessories  
- Medicine Bottles  

All images were annotated using **Roboflow**, a platform that simplifies dataset organization, preprocessing, and bounding-box annotation. Its interface ensures clean, consistent labeling across all classes, improving training quality and overall model performance.

---

## **📁 Repository Structure**

├── model/ # Final trained DQN model weights and artifacts
├── model_checkpoints/ # Intermediate model checkpoints during training
├── DQN_Model_training.ipynb # Notebook for preprocessing, environment setup, and DQN training
└── Object_detection_results_dqn.ipynb # Notebook for inference, visualization, and bounding box


---

## **🧩 System Design Overview**

### **1) Data Collection Module**
Images are sourced, organized, and prepared for annotation.  

### **2) Annotation Module**
Roboflow is used to draw bounding boxes and assign class labels with high precision.  
The platform’s annotation tools ensure consistent labeling across all nine medical waste categories.

### **3) Pre-processing and Augmentation Module**

**Pre-processing:**
- Auto-orientation correction  
- Resizing to a standard resolution (e.g., 654×654)  
- Grayscale conversion  
- Normalization of pixel values to the range 0–1  

**Augmentation:**
- Rotation  
- Horizontal/vertical flipping  
- Cropping  
- Shearing  
- Hue and saturation adjustments  
- Zooming  
- Bounding-box safe transformations  

### **4) Training Module (Deep Q-Network)**

The RL agent:
- Starts with an initial bounding box  
- Takes actions such as move, expand, shrink, or terminate  
- Receives rewards based on IoU improvement  
- Gradually converges to accurate bounding box predictions  
- Outputs class label and final bounding box  

Training configuration includes:
- Categorical Cross-Entropy loss  
- Adam optimizer  
- Metrics such as accuracy, precision, recall, and F1 score  
- Hyperparameter tuning for learning rate, batch size, and dropout  
- Replay buffer and epsilon-greedy exploration strategy  

### **5) Segregation Module**

After detection, the predicted class is linked to its respective biomedical waste color-coded bin:

- Red — Plastic Waste  
- Yellow — Infectious Waste  
- Blue — Sharp Waste  
- Green — Injurious Sharp Waste  
- Black — Cytotoxic Waste  

This final mapping supports healthcare waste management workflows.

---

## **📘 Notebook Descriptions**

### **DQN_Model_training.ipynb**
Contains the full workflow for:
- Loading annotated dataset  
- Applying preprocessing and augmentation  
- Designing the RL environment  
- Building the CNN-based Q-Network  
- Defining action space and reward mechanics  
- Running the DQN training loop  
- Saving model checkpoints and final weights  

### **Object_detection_results_dqn.ipynb**
Used for:
- Loading trained model from the `model/` directory  
- Running inference on unseen images  
- Visualizing predicted bounding boxes  
- Displaying scores, IoU, and class predictions  
- Mapping detected waste types to their disposal categories  

---

## **🚀 Future Enhancements**
- Multi-object detection using multi-agent RL  
- Integration with YOLOv8 for hybrid RL+DL detection  
- Real-time detection on embedded/edge devices  
- Support for multiple bounding boxes per frame  
- Increased dataset diversity for improved generalization

---
