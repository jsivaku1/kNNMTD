# kNNMTD: A No-Code Web Application for Synthetic Data Generation

## Overview
This repository provides a powerful, no-code web application for generating high-quality synthetic data from small or imbalanced datasets using the **k-Nearest Neighbor Mega-Trend Diffusion (kNNMTD)** algorithm. The algorithm is based on the research paper: ["Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors."](https://doi.org/10.1016/j.knosys.2021.107687).

The core of the algorithm uses a three-step iterative procedure to ensure the generated data preserves the statistical properties of the original dataset, even for minority classes:

1. For each data point, it identifies the **k-Nearest Neighbors (kNN)** within its own class (for classification tasks).
2. The neighboring samples are then used to create a plausible data range using **Mega-Trend Diffusion (MTD)**.
3. Finally, new samples are generated from these ranges and filtered to select the most realistic data points.

<div align="center">
  <br/>
  <p align="center">
    <img align="center" width="80%" src="https://github.com/jsivaku1/kNNMTD/blob/main/illustration.png" alt="kNNMTD Illustration"/>
  </p>
</div>

## ✨ Key Features of the Web Application

This repository has been updated from a simple script to a full-featured web application with an intelligent backend pipeline:

- **No-Code Interface:** An intuitive UI allows you to upload your dataset and generate synthetic data without writing a single line of code.
- **Automatic Data Cleaning Pipeline:** The app automatically handles common data issues:
  - **Missing Value Imputation:** Fills missing numerical data with the median and categorical data with the mode.
  - **Categorical Data Encoding:** Converts text-based columns into a numerical format for the algorithm.
- **Smart Task Detection:** You no longer need to specify the task type. The app automatically detects whether the task is **Classification**, **Regression**, or **Unsupervised** based on the target column you select.
- **Automatic `k` Optimization:** The application tests a range of `k` values to find the optimal one that produces the synthetic data with the best statistical similarity (lowest PCD score) to your original data.
- **Detailed Performance Analysis:** The results page provides a comprehensive analysis of the generated data, including:
  - A summary of the pipeline stages and the optimal `k` value chosen.
  - Plots showing the improvement of **PCD score** and **ML Utility metrics** (like Accuracy, F1-score, AUC) over each generation epoch.

## 🚀 How to Run the Web Application

### 1. Installation

First, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/jsivaku1/kNNMTD.git
cd kNNMTD
pip install -r requirements.txt
```

### 2. Running the App

Once the dependencies are installed, you can start the Flask web server with a single command:

```bash
python3 app.py
```

### 3. Usage

1. Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Upload your `.csv` dataset.
3. Select your target column (or leave it as "None" for unsupervised tasks).
4. Choose your desired parameters (`k` will be optimized automatically).
5. Click **Generate Data** and view the results.

---

## 📖 Citing kNNMTD

Please cite the original work if you are using this application or its underlying algorithm:

Jayanth Sivakumar, Karthik Ramamurthy, Menaka Radhakrishnan, and Daehan Won.  
"Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors."  
Knowledge-Based Systems (2021): 107687.

```bibtex
@article{sivakumar2021synthetic,
  title={Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors},
  author={Sivakumar, Jayanth and Ramamurthy, Karthik and Radhakrishnan, Menaka and Won, Daehan},
  journal={Knowledge-Based Systems},
  pages={107687},
  year={2021},
  publisher={Elsevier}
}
```