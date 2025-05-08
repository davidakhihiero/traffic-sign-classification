# Traffic Sign Classification for Autonomous Driving

This MATLAB project compares the effectiveness of different machine learning approaches (SVM + HOG, SVM + SIFT, and CNN) for classifying traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## 📁 Directory Structure

- `scripts/`: Contains all MATLAB scripts for training and testing models.
- `models/`: Saved `.mat` files of trained models (CNN, SVM-HOG, SVM-SIFT).
- `Data/`: Contains dataset.
- `DATASET_INSTRUCTIONS.txt`: Contains dataset download instructions.
- `report/`: Final project report in PDF format.

## 🧠 Models

- `SVM + HOG`: Good edge detection, ~92% accuracy.
- `SVM + SIFT`: Scale/rotation invariant, ~51% accuracy.
- `CNN`: Learns features automatically, ~95% accuracy.

## 🚀 How to Run

1. Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-classification.git
cd traffic-sign-classification/scripts
```

2. Launch MATLAB and open a script, e.g., `TrainTestTrafficSignsCNN.m`.

3. Make sure the GTSRB dataset is accessible in the expected path, or modify the scripts accordingly.

## 📊 Results

| Model         | Accuracy | Training Time | Testing Time |
|---------------|----------|---------------|---------------|
| SVM + HOG     | 92.20%   | 1186s         | 1924s         |
| SVM + SIFT    | 51.43%   | 2778s         | 1859s         |
| CNN           | 95.46%   | 211s          | 68s           |

## 📄 Report

Find detailed methodology, results, and analysis in [`report/David_Akhihiero_CS_677__Final_Project_Report.pdf`](./report/David_Akhihiero_CS_677__Final_Project_Report.pdf).

## 🧾 License

MIT License.

---

**Author**: David Akhihiero  
West Virginia University – CS 677 Final Project  
