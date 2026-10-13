import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(filename):
    # Read datasets from the project's data/ folder (or beside this script).
    base = Path(__file__).resolve().parent
    path = base.parent / 'data' / filename
    if not path.exists():
        path = base / filename
    data = np.load(path)
    return data['X'], data['Y'], data['X_test'], data['Y_test']

def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((X, ones))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def train_linear_regression(X, Y, use_bias=False):
    if use_bias:
        X = add_bias(X)
    # Normal Equation: w = (X^T X)^(-1) X^T Y
    XT = X.T
    try:
        w = np.linalg.solve(XT @ X, XT @ Y)
    except np.linalg.LinAlgError:
        # Rank-deficient dataset: ordinary inverse does not exist.
        w = np.linalg.pinv(X) @ Y
    return w

def evaluate_model(X, Y, w, use_bias=False):
    if use_bias:
        X = add_bias(X)
    y_pred = X @ w
    mse = mean_squared_error(Y, y_pred)
    return mse, y_pred

def analyze_affine2_perturbation(X, Y, X_test, Y_test, w, use_bias=True):
    np.random.seed(42)
    w_perturbed = w + 0.01 * np.random.randn(*w.shape)
    mse_train, _ = evaluate_model(X, Y, w_perturbed, use_bias)
    mse_test, _ = evaluate_model(X_test, Y_test, w_perturbed, use_bias)
    print("\n[affine2.npz] Perturbed weights:")
    print("MSE Train (perturbed):", mse_train)
    print("MSE Test (perturbed):", mse_test)

def analyze_zero(X, Y, X_test, Y_test, w, y_pred_train):
    print("\n[zero.npz] Weight coefficients:")
    print(w)
    print("Nearly zero weights might correspond to irrelevant features.")
    plt.figure(figsize=(8, 5))
    plt.plot(Y, label='True Output')
    plt.plot(y_pred_train, label='Predicted Output')
    plt.title("zero.npz: True vs Predicted Output")
    plt.legend()
    plt.show()

def lstsq_solution(X, Y):
    # Add bias explicitly
    X_bias = add_bias(X)
    w_lstsq, residuals, rank, s = np.linalg.lstsq(X_bias, Y, rcond=None)
    return w_lstsq, residuals, rank, s

def solve_and_print(filename, use_bias=False, analyze=False):
    print(f"\n--- Solving: {filename} ---")
    X, Y, X_test, Y_test = load_data(filename)

    w = train_linear_regression(X, Y, use_bias=use_bias)
    mse_train, y_pred_train = evaluate_model(X, Y, w, use_bias)
    mse_test, y_pred_test = evaluate_model(X_test, Y_test, w, use_bias)

    print("Weights (including bias if used):", w)
    print("Training MSE:", mse_train)
    print("Test MSE:", mse_test)

    if analyze and filename == 'affine2.npz':
        analyze_affine2_perturbation(X, Y, X_test, Y_test, w, use_bias=True)

    if analyze and filename == 'zero.npz':
        analyze_zero(X, Y, X_test, Y_test, w, y_pred_train)

    if analyze and filename == 'rank.npz':
        print("\nUsing np.linalg.pinv and lstsq for rank.npz:")
        X_bias = add_bias(X)
        w_pinv = np.linalg.pinv(X_bias) @ Y
        w_lstsq, residuals, rank, s = lstsq_solution(X, Y)
        mse_pinv, _ = evaluate_model(X_test, Y_test, w_pinv, use_bias=True)
        mse_lstsq, _ = evaluate_model(X_test, Y_test, w_lstsq, use_bias=True)
        print("MSE (pinv):", mse_pinv)
        print("MSE (lstsq):", mse_lstsq)
        print("lstsq rank:", rank)
        print("Singular values:", s)

# === Run all datasets ===
if __name__ == '__main__':
    solve_and_print('linear.npz', use_bias=False)
    solve_and_print('affine1.npz', use_bias=True)
    solve_and_print('affine2.npz', use_bias=True, analyze=True)
    solve_and_print('rank.npz', use_bias=True, analyze=True)
    solve_and_print('zero.npz', use_bias=True, analyze=True)
