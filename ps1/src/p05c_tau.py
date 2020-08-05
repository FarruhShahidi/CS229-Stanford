import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    best_tau, best_mse = None, float('inf')
    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau)

        lwr.fit(x_train, y_train)

        x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
        y_pred = lwr.predict(x_val)
        mse = ((y_pred - y_val) ** 2).mean()
        if mse < best_mse:
            best_tau = tau
        best_mse = min(mse, best_mse)

    # Fit a LWR model with the best tau value
    lwr = LocallyWeightedLinearRegression(best_tau)
    lwr.fit(x_train, y_train)
    print("best tau is: {}, best mse is : {}".format(best_tau, best_mse))
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_test_pred = lwr.predict(x_test)
    mse = ((y_test_pred - y_test) ** 2).mean()
    print("mse for the test set : {}".format(mse))
    # Save predictions to pred_path
    np.savetxt(pred_path, y_test_pred)
    # Plot data
    plt.figure()
    
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_test, y_test_pred, 'ro')
    plt.savefig("output/lwr.png")
    # *** END CODE HERE ***
