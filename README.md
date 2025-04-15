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
   conda create -n FAMyoS python=3.9 #python version <= 3.9
   conda activate FAMyoSconda  
   ```

3. **Install the required dependencies:**
   Install [Pytorch](https://pytorch.org/get-started/locally/) (<= 2.6.0) and other packages. 

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
python run_inference.py --input /path/to/your/images --output /path/to/save/results
```

Make sure the model weights are downloaded and the path is correctly set within the code or configuration file.

---

## 📖 Citation

If you use FAMyoS in your research, please cite the corresponding paper:

> _[Add your citation here, e.g., journal article, conference paper, or preprint]_

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

FAMyoS was developed as part of ongoing research in cardiac image analysis and deep learning.  
Special thanks to all contributors and collaborators!
