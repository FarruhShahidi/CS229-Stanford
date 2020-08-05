import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    gda = GDA()
    gda.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = gda.predict(x_eval)
    # Plot decision boundary on validation set
    util.plot(x_eval, y_eval, gda.theta, "{}.png".format(pred_path))

    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        m, n = x.shape
        phi = np.mean(y) # could also use (y == 0).sum()
        mu_0 = x[y == 0].sum(axis = 0)/(y == 0).sum()
        mu_1 = x[y == 1].sum(axis = 0)/(y == 1).sum()
        w = x.copy()
        w[y == 0] -= mu_0
        w[y == 1] -= mu_1

        sigma = (1/m) * w.T.dot(w)
        sigma_inv = np.linalg.inv(sigma)
        theta = sigma_inv.T.dot(mu_1 - mu_0)
        theta_0 = (1/2) * (sigma_inv.dot(mu_0).T.dot(mu_0) -
                            sigma_inv.dot(mu_1).T.dot(mu_1))\
                            - np.log((1 - phi)/phi)
        theta = np.hstack([np.array([theta_0]), theta])
        self.theta = theta
        return theta







        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        # define the logistic function
        def g(z):
            return 1 / (1 + np.exp(-z))
        #x = util.add_intercept(x)
        probs = g(x.dot(self.theta))
        y_pred = (probs >= 0.5).astype(np.int)


        return y_pred

        # *** END CODE HERE
