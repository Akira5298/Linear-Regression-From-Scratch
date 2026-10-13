# Linear Regression from Scratch

A Python project exploring linear regression using NumPy matrix operations, with experiments on five datasets and visualizations using Matplotlib.

## Features

- Train linear regression models with or without a bias term.
- Evaluate predictions using training and test Mean Squared Error (MSE).
- Compare original and slightly perturbed model weights.
- Handle rank-deficient data using the pseudoinverse and compare it with NumPy's least-squares solver.
- Plot actual versus predicted outputs for the `zero.npz` dataset.

## Project Structure

```text
linear-regression-from-scratch/
├── src/
│   └── linear_regression.py
├── data/
│   ├── linear.npz
│   ├── affine1.npz
│   ├── affine2.npz
│   ├── rank.npz
│   └── zero.npz
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

Install the dependencies from the repository's root folder:

```bash
python3 -m pip install -r requirements.txt
```

Run all experiments:

```bash
python3 src/linear_regression.py
```

The script prints the training and test MSE for each dataset and displays a plot for `zero.npz`.

## Technologies

Python · NumPy · Matplotlib
