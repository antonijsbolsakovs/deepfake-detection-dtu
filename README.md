# Deepfake Detection Using AI

Bachelor thesis project (BSc, DTU) on detecting deepfake images using deep learning.  
The project focuses on classifying static facial images as **real** or **fake** with AI models.

## 📄 Project Overview
Deepfake technology is rapidly evolving, making fake images harder to detect.  
This project explores several approaches for image-based detection:
- A **Custom CNN** built from scratch as a baseline.
- Fine-tuned **XceptionNet**.
- Fine-tuned **EfficientNet-B0**.

The dataset was derived from **FaceForensics++**, extracting and balancing real vs. fake images:
- **Train:** 8,000 real + 8,000 fake  
- **Test:** 2,000 real + 2,000 fake  

Additional evaluation included:
- Accuracy, precision, recall, F1-score  
- Confusion matrices  
- Grad-CAM visualizations for interpretability  

📑 Full thesis report: [`Deepfake_Detection_BSc_Thesis.pdf`](./Deepfake_Detection_BSc_Thesis.pdf)

---

## 📂 Repository Structure
```plaintext
├── src/                  # Source code (training, evaluation, preprocessing)
├── results/              # Model checkpoints, metrics, Grad-CAM visualizations
├── Deepfake_Detection_BSc_Thesis.pdf  
├── requirements.txt      # Dependencies
├── LICENSE
├── .gitignore
└── README.md
````` 

---

## 🚀 Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/username/deepfake-detection.git
   cd deepfake-detection
   ````` 

2.	Install dependencies:
    ```bash
    pip install -r requirements.txt
    ````` 
4.	Run training (example for custom CNN):
    ```bash
    python src/train_custom_cnn.py
    ````` 

6.	Evaluate a trained model:
    ```bash
    python src/evaluate_custom_cnn.
    `````
## 💾 Pretrained Models

The best-performing models obtained during training have been saved as PyTorch checkpoint files and are available on Google Drive:

- [Custom CNN](https://drive.google.com/file/d/1N2XTW8PUU7YeckAy-sQY5kC7TbjO4Y47/view)  
- [EfficientNet-B0](https://drive.google.com/file/d/1JnpQdoulryDO3rxqXbZX-vDOER2SrZqa/view)  
- [XceptionNet](https://drive.google.com/file/d/13Ccev-9RJlDqw2eTJcvLuoieCMDijp-v/view)  

These files contain all the learned model parameters and can be loaded using the evaluation scripts in `src/` to reproduce the results presented in the thesis report.
    
## 📊 Results (Summary)

| Model        | Validation Accuracy | Test Accuracy |
|--------------|----------------------|---------------|
| Custom CNN   | 81.2%               | 79.3%         |
| EfficientNet-B0 | 96.6%            | 96.6%         |
| XceptionNet  | 98.1%               | 98.0%         |

All three models performed well in detecting deepfakes. The Custom CNN achieved solid performance despite being lightweight and trained from scratch, proving the feasibility of simpler architectures. Grad-CAM visualizations show that while each model identified relevant facial regions, XceptionNet provided the most precise and consistent focus on manipulations.

---

## 🔮 Future Work
- Extend detection to **video-based** deepfakes.  
- Train **deeper custom CNNs** with longer training schedules.  
- Use larger and more diverse datasets (e.g., **DFDC**, **Celeb-DF**).  

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.  

---

## 👤 Author
**Antonijs Bolsakovs**  
BSc in General Engineering (Cyber Systems), Technical University of Denmark (DTU)  
