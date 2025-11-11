import numpy as np
import os

# ---------------------------------------------------------
# Load a single PGM file (binary P5 format)
# ---------------------------------------------------------
def load_pgm_bin(filename):
    """
    Load a PGM image in binary (P5) format.
    Returns a numpy array of shape (height, width).
    """
    with open(filename, 'rb') as f:
        # Read magic number
        magic = f.readline()
        if magic.strip() != b'P5':
            raise ValueError(f"{filename} is not a P5 PGM file")

        # Skip comments if present
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        # Read image size
        width, height = map(int, line.strip().split())

        # Read maximum pixel value
        maxval = int(f.readline().strip())
        if maxval > 255:
            raise ValueError("Only 8-bit PGM files are supported")

        # Read binary pixel data
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
        img = img_data.reshape((height, width))
        return img


# ---------------------------------------------------------
# Load MNIST dataset from a single folder
# ---------------------------------------------------------
def load_mnist_pgm_dataset(folder):
    """
    Loads all .pgm files from a given folder.
    Each filename is expected to have the format '<digit>_xxxxx.pgm'.
    Returns:
        X: array of flattened images normalized to [0,1]
        y: array of integer class labels
    """
    X, y = [], []
    for filename in os.listdir(folder):
        if filename.endswith('.pgm') and '_' in filename:
            label = int(filename.split('_')[0])  # Extract digit before underscore
            img = load_pgm_bin(os.path.join(folder, filename))
            X.append(img.flatten())
            y.append(label)
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=int)
    return X, y


# ---------------------------------------------------------
# Linear SVM (One-vs-Rest) training from scratch
# ---------------------------------------------------------
def train_svm_ovr(X, y, n_classes=10, lr=0.001, lambda_param=0.01, n_iters=5):
    """
    Trains a one-vs-rest linear SVM using stochastic gradient descent.
    """
    n_samples, n_features = X.shape
    W = np.zeros((n_classes, n_features))
    b = np.zeros(n_classes)

    for c in range(n_classes):
        print(f"Training class {c} vs all...")
        y_binary = np.where(y == c, 1, -1)
        w = np.zeros(n_features)
        bias = 0

        for epoch in range(n_iters):
            for i in range(n_samples):
                x_i = X[i]
                condition = y_binary[i] * (np.dot(x_i, w) + bias) >= 1
                if condition:
                    w -= lr * (2 * lambda_param * w)
                else:
                    w -= lr * (2 * lambda_param * w - y_binary[i] * x_i)
                    bias += lr * y_binary[i]

        W[c, :] = w
        b[c] = bias
        print(f"  Done training class {c}.")
    return W, b


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_svm_ovr(X, W, b):
    """
    Predicts class labels using trained SVM weights and biases.
    """
    scores = np.dot(X, W.T) + b
    return np.argmax(scores, axis=1)


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Path to folder containing all MNIST .pgm files
    data_folder = "mnist_s"
    train_data_folder = "yymnist-master/mnist/train"
    test_data_folder = "yymnist-master/mnist/test"

    print("Loading MNIST PGM files...")
    X_train, y_train = load_mnist_pgm_dataset(train_data_folder)
    print(f"Loaded {len(X_train)} train images, each of dimension {X_train.shape[1]} features.")
    X_test, y_test = load_mnist_pgm_dataset(test_data_folder)
    print(f"Loaded {len(X_test)} test images, each of dimension {X_test.shape[1]} features.")

    print("Training SVM classifier...")
    W, b = train_svm_ovr(X_train, y_train, n_classes=10, lr=0.001, lambda_param=0.01, n_iters=5)

    print("Evaluating...")
    y_pred = predict_svm_ovr(X_test, W, b)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
