# Pixel-Coordinate-Regression
CNN based supervised regression system to accurately localize a single bright pixel in 50×50 grayscale images. The project uses synthetic data generation, deep learning modeling, training visualization, and quantitative evaluation using MAE and MSE metrics.

Pixel Coordinate Regression using Convolutional Neural Networks (CNN)

1. Project Overview

This project implements a deep learning-based pixel localization system that predicts the (x, y) coordinates of a bright pixel inside a grayscale image using a Convolutional Neural Network (CNN).

The task is formulated as a supervised regression problem, where the model learns a direct mapping from image pixel values to continuous coordinate outputs.

A synthetic dataset is generated to ensure precise ground truth labeling, controlled experimentation and reproducibility.


2. Objectives

Generate a synthetic image dataset with known pixel coordinates.

Design a CNN-based regression model to predict pixel location.

Train and evaluate the model using MSE, MAE, and RMSE metrics.

Visualize predictions and learning curves.

Ensure robustness, clarity, reproducibility, and maintainability.


3. Why CNN for This Task?

Convolutional Neural Networks are ideal for this problem because:

They automatically learn spatial features from images.

They capture local pixel patterns and spatial hierarchies.

They generalize well to unseen pixel locations.

They efficiently map 2D image inputs → continuous outputs.

CNNs eliminate the need for handcrafted features and provide superior learning performance for visual regression tasks.


4. Dataset Description
Synthetic Dataset Generation

Each generated image:

Size: 50 × 50 pixels

Contains exactly one bright pixel (value = 255)

All other pixels are zero

Ground truth label = (x, y) coordinate of the bright pixel

Why Synthetic Dataset?

Using a synthetic dataset ensures:

Perfect labeling accuracy (no annotation noise)

Controlled data distribution

Balanced spatial coverage

Full reproducibility

Eliminates bias introduced by real-world datasets

This allows the focus to remain on model design, training stability, and evaluation methodology, rather than data quality issues.

5. Data Preprocessing Pipeline

Pixel normalization: [0, 255] → [0, 1]

Coordinate normalization: [0, 49] → [0, 1]

Train-validation-test split: 70% / 15% / 15%

Optional augmentation:

Gaussian noise

Gaussian blur

Blob-based spatial smoothing

These steps improve:

Training stability

Model generalization

Robustness against noise


6. Model Architecture

The model is a CNN-based regressor that predicts continuous pixel coordinates.

Architecture Summary

<img width="324" height="657" alt="image" src="https://github.com/user-attachments/assets/499f3f3e-8cbf-4332-aa0c-dee917d95c7c" />





Key Design Choices

Global Average Pooling reduces parameters and prevents overfitting.

Sigmoid output layer ensures predictions stay in [0,1].

Dropout layers improve generalization.

MSE loss optimized for regression accuracy.


7. Training Strategy

Optimizer: Adam (1e-4 learning rate)

Loss Function: Mean Squared Error (MSE)

i. Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

ii. Callbacks:

EarlyStopping

ReduceLROnPlateau


8.Model Performance

Final Metrics

<img width="592" height="166" alt="image" src="https://github.com/user-attachments/assets/5521d67f-4e2b-4e49-ab04-7ccf1990d5b2" />



Interpretation

The model achieves sub-3 pixel localization error.

Training, validation, and test errors are consistent, indicating:

No overfitting

Strong generalization

This accuracy is excellent for pixel regression tasks.


9. Visual Results

The project includes:

Training vs validation loss curves

RMSE learning curves

Visual overlays of:

Ground truth coordinates

Predicted coordinates

These visualizations confirm:

Stable convergence

Accurate localization

Strong spatial learning

10. Computational Efficiency

Training time: ~3–6 minutes on standard GPU

Inference time: ~2–3 ms per image

Suitable for real-time localization applications

11. Installation Instructions
Requirements

Python ≥ 3.9

Setup
pip install -r requirements.txt
Run Notebook
jupyter notebook

Open the provided .ipynb file and run all cells.

12. Reproducibility

Fixed random seeds ensure consistent outputs

Fully synthetic dataset enables deterministic results

Modular and documented code structure ensures maintainability

13. Project Applications

Object localization

Landmark detection

Medical image analysis

Robotics vision

Autonomous navigation

Industrial inspection systems

14. Final Conclusion

This project demonstrates a complete deep learning pipeline — from synthetic data generation to model training, evaluation, and visualization.

The CNN-based regression approach achieves high accuracy, robustness, and generalization, making it suitable for real-world coordinate prediction tasks.

The structured design, strong evaluation metrics, and detailed analysis reflect industry-grade machine learning practices.
