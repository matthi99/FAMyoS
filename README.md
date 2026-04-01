# FAMyoS
**Deep Learning Pipeline for Fully Automated Myocardial Infarct Segmentation**

FAMyoS is a deep learning-based pipeline designed to automatically segment myocardial infarcts from cardiac MRI images. The system utilizes a trained neural network to deliver accurate and reproducible segmentation results with minimal user intervention.

---

## 🚀 Features

- ✅ Fully automated segmentation of myocardial infarcts  
- 🧠 State-of-the-art deep learning architecture  
- 🖥️ Easy-to-use pipeline for both research and clinical data  
- 📦 Pretrained model weights for immediate use

---

## 🛠 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/matthi99/FAMyoS.git
   cd FAMyoS
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   conda create -n FAMyoS python=3.9 
   conda activate FAMyoS
   ```

3. **Install the required dependencies:**
   - Install [Pytorch](https://pytorch.org/get-started/locally/) (Version <= 2.6.0)
   - Install other packages
   ```bash
   pip install -r requirements.txt
   ```

---

## ⬇️ Download Model Weights

The pretrained model weights are not included in this repository.  
Please download them from the following link:

👉 [Download FAMyoS Weights on Google Drive](https://drive.google.com/drive/folders/1_UiK4XLT5Kt7HkpfQOkafls_eshtad-d?usp=sharing)

Once downloaded, place the weights into a folder named `weights/` or the directory expected by your configuration.

---

## 🧪 Usage

Run the segmentation on your own data:

```bash
python inference.py --patient_folder /path/to/your/images --save_folder /path/to/save/results --plots True/False 
```
- `--patient_folder` specifies the path to your dicom images (`default="dicoms/"`). Data should be saved as dicom files in folders called Patient_1, Patient_2, and so on.
- `--save_folder` path were the results should get saved (`default="segmentations/"`).
- `--plots` If True png file get saved visualizing segmentation results (`default=False`). 


Matthias Schwab

<sup>1</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria
