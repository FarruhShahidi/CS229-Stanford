import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # define the model
    clf = LogisticRegression()

    # Train a logistic regression classifier
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    # Plot decision boundary on top of validation set set
    util.plot(x_eval, y_eval, clf.theta, "{}.png".format(pred_path))
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    # define the logistic function
    def g(self, z):
        return 1 / (1 + np.exp(-z))


    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # get the parameters m and n
        m, n = x.shape
        # initialize vector theta to zero
        if not self.theta:
            self.theta = np.zeros(n)


        while True:
            theta = self.theta
            # for each training example x^(i) compute (theta^T, x^(i))
            # and store as a vector.
            theta_t_x = x.dot(theta)
            # compute the gradient according to part (a)
            J_grad = (- 1/ m) * (y - self.g(theta_t_x)).dot(x)
            # define the Hessian and its inverse
            # the formula for H is derived in part a.
            H = (1 / m) * self.g(theta_t_x).dot(1 - self.g(theta_t_x)) * (x.T).dot(x)
            # get the inverse Hessian
            H_inv = np.linalg.inv(H)
            # start iteration
            theta_next = theta - np.matmul(H_inv, J_grad)
            # if theta's close enough then stop
            if np.linalg.norm(theta_next - theta, ord=1) < self.eps or self.max_iter == 0:
                break
            self.theta = theta_next
            self.max_iter -= 1


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        probs = self.g(x.dot(self.theta))
        y_pred = (probs >= 0.5).astype(np.int)

        return y_pred
        # *** END CODE HERE ***
