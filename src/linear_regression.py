import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Load Dataset
# =========================
def load_data(filename):
    base = Path(__file__).resolve().parent
    path = base.parent / 'data' / filename

    if not path.exists():
        path = base / filename

    data = np.load(path)
    return data['X'], data['Y'], data['X_test'], data['Y_test']


# =========================
# Linear Regression Helpers
# =========================
def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((X, ones))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def train_linear_regression(X, Y, use_bias=False):
    if use_bias:
        X = add_bias(X)

    XT = X.T
    try:
        w = np.linalg.solve(XT @ X, XT @ Y)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(X) @ Y

    return w

def evaluate_model(X, Y, w, use_bias=False):
    if use_bias:
        X = add_bias(X)

    y_pred = X @ w
    mse = mean_squared_error(Y, y_pred)
    return mse, y_pred


# =========================
# Plotting
# =========================
def plot_single_output(y_true, y_pred, filename, dataset_type, save_plot=False):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.plot(y_true, label='True Output', color='blue')
    ax.plot(y_pred, label='Predicted Output', color='red', linestyle='--')

    mse = mean_squared_error(y_true, y_pred)

    ax.set_title(f"{filename}: True vs Predicted ({dataset_type})")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Output Value")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Set different window title instead of all being "Figure 1"
    try:
        fig.canvas.manager.set_window_title(f"{filename} - {dataset_type}")
    except Exception:
        pass

    # Save with unique filename if requested
    if save_plot:
        output_dir = Path(__file__).resolve().parent / "plots"
        output_dir.mkdir(exist_ok=True)

        safe_name = filename.replace(".npz", "")
        out_path = output_dir / f"{safe_name}_{dataset_type.lower()}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {out_path}")

    plt.show()


# =========================
# Special Analyses
# =========================
def analyze_affine2_perturbation(X, Y, X_test, Y_test, w, use_bias=True):
    np.random.seed(42)
    w_perturbed = w + 0.01 * np.random.randn(*w.shape)

    mse_train, _ = evaluate_model(X, Y, w_perturbed, use_bias)
    mse_test, _ = evaluate_model(X_test, Y_test, w_perturbed, use_bias)

    print("\n[affine2.npz] Perturbed weights:")
    print(w_perturbed)
    print("MSE Train (perturbed):", mse_train)
    print("MSE Test (perturbed):", mse_test)

def analyze_zero(w):
    print("\n[zero.npz] Weight coefficients:")
    print(w)
    print("Nearly zero weights might correspond to irrelevant features.")

def lstsq_solution(X, Y):
    X_bias = add_bias(X)
    w_lstsq, residuals, rank, s = np.linalg.lstsq(X_bias, Y, rcond=None)
    return w_lstsq, residuals, rank, s


# =========================
# Main Solver
# =========================
def solve_and_print(filename, use_bias=False, analyze=False,
                    show_train_plot=False, show_test_plot=True, save_plots=False):
    print(f"\n{'=' * 60}")
    print(f"Solving: {filename}")
    print(f"{'=' * 60}")

    X, Y, X_test, Y_test = load_data(filename)

    w = train_linear_regression(X, Y, use_bias=use_bias)

    mse_train, y_pred_train = evaluate_model(X, Y, w, use_bias)
    mse_test, y_pred_test = evaluate_model(X_test, Y_test, w, use_bias)

    print("Weights (including bias if used):")
    print(w)

    print("\nTraining MSE:", mse_train)
    print("Test MSE:", mse_test)

    if analyze and filename == 'affine2.npz':
        analyze_affine2_perturbation(X, Y, X_test, Y_test, w, use_bias=True)

    if analyze and filename == 'zero.npz':
        analyze_zero(w)

    if analyze and filename == 'rank.npz':
        print("\n[rank.npz] Using np.linalg.pinv and np.linalg.lstsq:")
        X_bias = add_bias(X)

        w_pinv = np.linalg.pinv(X_bias) @ Y
        w_lstsq, residuals, rank, s = lstsq_solution(X, Y)

        mse_pinv, _ = evaluate_model(X_test, Y_test, w_pinv, use_bias=True)
        mse_lstsq, _ = evaluate_model(X_test, Y_test, w_lstsq, use_bias=True)

        print("MSE (pinv):", mse_pinv)
        print("MSE (lstsq):", mse_lstsq)
        print("lstsq rank:", rank)
        print("Singular values:", s)

    if show_train_plot:
        plot_single_output(
            Y,
            y_pred_train,
            filename,
            dataset_type="Training",
            save_plot=save_plots
        )

    if show_test_plot:
        plot_single_output(
            Y_test,
            y_pred_test,
            filename,
            dataset_type="Test",
            save_plot=save_plots
        )


# =========================
# Menu Helpers
# =========================
def dataset_menu():
    print("\n=== Choose Dataset ===")
    print("1. linear.npz")
    print("2. affine1.npz")
    print("3. affine2.npz")
    print("4. rank.npz")
    print("5. zero.npz")
    print("0. Back")

def output_menu():
    print("\n=== Choose Output Type ===")
    print("1. Show training output only")
    print("2. Show test output only")
    print("3. Show both training and test outputs")
    print("4. Run all datasets (training only)")
    print("5. Run all datasets (test only)")
    print("6. Run all datasets (both training and test)")
    print("0. Exit")

def get_dataset_info(choice):
    datasets = {
        '1': ('linear.npz', False, False),
        '2': ('affine1.npz', True, False),
        '3': ('affine2.npz', True, True),
        '4': ('rank.npz', True, True),
        '5': ('zero.npz', True, True)
    }
    return datasets.get(choice, None)


# =========================
# Interactive Program
# =========================
if __name__ == '__main__':
    while True:
        output_menu()
        main_choice = input("\nSelect an option: ").strip()

        if main_choice == '0':
            print("Exiting...")
            break

        elif main_choice in ['1', '2', '3']:
            dataset_menu()
            d_choice = input("\nSelect a dataset: ").strip()

            if d_choice == '0':
                continue

            dataset_info = get_dataset_info(d_choice)
            if dataset_info is None:
                print("Invalid dataset selection.")
                continue

            filename, use_bias, analyze = dataset_info

            save_choice = input("Save plot image file as well? (y/n): ").strip().lower()
            save_plots = (save_choice == 'y')

            if main_choice == '1':
                solve_and_print(
                    filename,
                    use_bias=use_bias,
                    analyze=analyze,
                    show_train_plot=True,
                    show_test_plot=False,
                    save_plots=save_plots
                )
            elif main_choice == '2':
                solve_and_print(
                    filename,
                    use_bias=use_bias,
                    analyze=analyze,
                    show_train_plot=False,
                    show_test_plot=True,
                    save_plots=save_plots
                )
            elif main_choice == '3':
                solve_and_print(
                    filename,
                    use_bias=use_bias,
                    analyze=analyze,
                    show_train_plot=True,
                    show_test_plot=True,
                    save_plots=save_plots
                )

        elif main_choice in ['4', '5', '6']:
            save_choice = input("Save plot image files as well? (y/n): ").strip().lower()
            save_plots = (save_choice == 'y')

            all_datasets = {
                '1': ('linear.npz', False, False),
                '2': ('affine1.npz', True, False),
                '3': ('affine2.npz', True, True),
                '4': ('rank.npz', True, True),
                '5': ('zero.npz', True, True)
            }

            for filename, use_bias, analyze in all_datasets.values():
                if main_choice == '4':
                    solve_and_print(
                        filename,
                        use_bias=use_bias,
                        analyze=analyze,
                        show_train_plot=True,
                        show_test_plot=False,
                        save_plots=save_plots
                    )
                elif main_choice == '5':
                    solve_and_print(
                        filename,
                        use_bias=use_bias,
                        analyze=analyze,
                        show_train_plot=False,
                        show_test_plot=True,
                        save_plots=save_plots
                    )
                elif main_choice == '6':
                    solve_and_print(
                        filename,
                        use_bias=use_bias,
                        analyze=analyze,
                        show_train_plot=True,
                        show_test_plot=True,
                        save_plots=save_plots
                    )

        else:
            print("Invalid option. Please try again.")
