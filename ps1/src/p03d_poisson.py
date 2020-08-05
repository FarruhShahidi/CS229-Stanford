import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    poisson_model = PoissonRegression(step_size=lr, max_iter=1000)
    # Fit a Poisson Regression model
    poisson_model.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_val, y_val = util.load_dataset(eval_path)
    y_pred = poisson_model.predict(x_val)
    #util.plot(x_val, y_val, poisson_model.theta,  '{}.png'.format(pred_path))
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if not self.theta:
            self.theta = np.zeros(n)

        for _ in range(self.max_iter):
            theta = self.theta
            # compute theta^T x for each training example
            eta = x.dot(theta)
            # evaluate the function a
            a = np.exp(eta)
            # update theta
            theta_next = theta + (1/m) * self.step_size * (y - a).dot(x)
            if np.linalg.norm(theta_next - theta, ord=1) < self.eps:
                break

            self.theta = theta_next

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***
