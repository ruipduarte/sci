import os
import math
import random


# ==========================================================
#                 PGM (P5) FILE LOADER
# ==========================================================
def load_pgm_p5(filename):
    """
    Loads a P5 PGM file (binary) and returns a 28x28 list of lists,
    with values normalized to [0,1].
    """
    with open(filename, "rb") as f:
        # Read magic number
        assert f.readline().strip() == b'P5'
        # Skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        w, h = map(int, line.split())
        maxval = int(f.readline())
        assert w == 28 and h == 28

        data = list(f.read(w * h))
        # Convert to 28×28 matrix normalized to [0,1]
        img = []
        for i in range(28):
            row = [data[i*28 + j] / maxval for j in range(28)]
            img.append(row)
        return img


def load_dataset(directory):
    images = []
    labels = []
    for fname in os.listdir(directory):
        if not fname.endswith(".pgm"):
            continue
        label = int(fname.split("_")[0])   # label from filename
        img = load_pgm_p5(os.path.join(directory, fname))
        images.append(img)
        labels.append(label)
    return images, labels


# ==========================================================
#                CNN LAYERS: DISCRETE IMPLEMENTATION
# ==========================================================

def conv2d(image, kernel):
    H = len(image)
    W = len(image[0])
    k = len(kernel)
    outH = H - k + 1
    outW = W - k + 1
    out = [[0]*outW for _ in range(outH)]
    for i in range(outH):
        for j in range(outW):
            s = 0.0
            for ki in range(k):
                for kj in range(k):
                    s += image[i+ki][j+kj] * kernel[ki][kj]
            out[i][j] = s
    return out


def relu(feature_map):
    return [[max(0, v) for v in row] for row in feature_map]


def maxpool2x2(feature_map):
    H = len(feature_map)
    W = len(feature_map[0])
    out = []
    for i in range(0, H, 2):
        row = []
        for j in range(0, W, 2):
            block = [
                feature_map[i][j],
                feature_map[i][j+1],
                feature_map[i+1][j],
                feature_map[i+1][j+1]
            ]
            row.append(max(block))
        out.append(row)
    return out


def fc_layer(vec, weights, biases):
    out = []
    for i in range(len(weights)):
        s = biases[i]
        for j in range(len(vec)):
            s += weights[i][j] * vec[j]
        out.append(s)
    return out


def softmax(logits):
    exps = [math.exp(v) for v in logits]
    s = sum(exps)
    return [v/s for v in exps]


# ==========================================================
#                    LOSS + BACKPROP
# ==========================================================
def cross_entropy(pred, label):
    return -math.log(pred[label] + 1e-12)


def argmax(arr):
    max_val = arr[0]
    idx = 0
    for i, v in enumerate(arr):
        if v > max_val:
            max_val = v
            idx = i
    return idx


# ==========================================================
#                 FORWARD PASS (for training)
# ==========================================================
def cnn_forward(image, kernel, fc_w, fc_b):
    conv = conv2d(image, kernel)
    rel = relu(conv)
    pool = maxpool2x2(rel)
    flat = [v for row in pool for v in row]
    logits = fc_layer(flat, fc_w, fc_b)
    probs = softmax(logits)
    return conv, rel, pool, flat, logits, probs


# ==========================================================
#                 TRAINING BACKPROP (minimal)
# ==========================================================
def train_one(image, label, kernel, fc_w, fc_b, lr):
    # Forward
    conv, rel, pool, flat, logits, probs = cnn_forward(image, kernel, fc_w, fc_b)

    # --------- Gradients for softmax + cross entropy ----------
    dlogits = probs[:]
    dlogits[label] -= 1.0  # derivative of softmax-crossentropy

    # --------- Gradients for FC layer ----------
    # FC weight gradients
    d_fc_w = []
    for i in range(len(fc_w)):
        row = []
        for j in range(len(fc_w[i])):
            row.append(dlogits[i] * flat[j])
        d_fc_w.append(row)

    # FC bias gradients
    d_fc_b = dlogits[:]

    # Backprop into flat
    dflat = [0.0]*len(flat)
    for i in range(len(fc_w)):
        for j in range(len(fc_w[i])):
            dflat[j] += dlogits[i] * fc_w[i][j]

    # --------- Unflatten to pool shape ----------
    H = len(pool)
    W = len(pool[0])
    dpool = [[0.0]*W for _ in range(H)]
    idx = 0
    for i in range(H):
        for j in range(W):
            dpool[i][j] = dflat[idx]
            idx += 1

    # --------- Backprop through maxpool 2x2 ----------
    drel = [[0.0]*(W*2) for _ in range(H*2)]
    for i in range(H):
        for j in range(W):
            val = pool[i][j]
            # find the location that produced the max
            base_i = i*2
            base_j = j*2
            block = [
                (base_i, base_j),
                (base_i, base_j+1),
                (base_i+1, base_j),
                (base_i+1, base_j+1)
            ]
            # distribute gradient only to max location
            maxloc = max(block, key=lambda pos: rel[pos[0]][pos[1]])
            drel[maxloc[0]][maxloc[1]] += dpool[i][j]

    # --------- Backprop through ReLU ----------
    dconv = [[0.0]*len(rel[0]) for _ in range(len(rel))]
    for i in range(len(rel)):
        for j in range(len(rel[0])):
            dconv[i][j] = drel[i][j] * (1 if rel[i][j] > 0 else 0)

    # --------- Backprop through convolution ----------
    dk = [[0.0]*3 for _ in range(3)]
    for i in range(len(dconv)):
        for j in range(len(dconv[0])):
            for ki in range(3):
                for kj in range(3):
                    dk[ki][kj] += dconv[i][j] * image[i+ki][j+kj]

    # --------- Update parameters ----------
    # Conv kernel update
    for i in range(3):
        for j in range(3):
            kernel[i][j] -= lr * dk[i][j]

    # FC update
    for i in range(len(fc_w)):
        for j in range(len(fc_w[i])):
            fc_w[i][j] -= lr * d_fc_w[i][j]
        fc_b[i] -= lr * d_fc_b[i]

    loss = cross_entropy(probs, label)
    return loss


# ==========================================================
#           SAVE + LOAD FUNCTIONS (TEXT FORMAT)
# ==========================================================
def save_conv_kernel(kernel, filename):
    with open(filename, "w") as f:
        for row in kernel:
            f.write(",".join(str(v) for v in row) + "\n")


def save_fc_weights(weights, filename):
    with open(filename, "w") as f:
        for row in weights:
            f.write(",".join(str(v) for v in row) + "\n")


def save_fc_biases(biases, filename):
    with open(filename, "w") as f:
        for b in biases:
            f.write(str(b) + "\n")


def load_conv_kernel(filename):
    k = []
    with open(filename, "r") as f:
        for line in f:
            k.append([float(x) for x in line.strip().split(",")])
    return k


def load_fc_weights(filename):
    w = []
    with open(filename, "r") as f:
        for line in f:
            w.append([float(x) for x in line.strip().split(",")])
    return w


def load_fc_biases(filename):
    b = []
    with open(filename, "r") as f:
        for line in f:
            b.append(float(line.strip()))
    return b


# ==========================================================
#                      MAIN TRAINING LOOP
# ==========================================================
def train_cnn(train_dir, test_dir, epochs=1, lr=0.01):
    # Load data
    train_images, train_labels = load_dataset(train_dir)
    test_images, test_labels = load_dataset(test_dir)

    # Initialize parameters
    kernel = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(3)]
    fc_w = [[random.uniform(-0.1, 0.1) for _ in range(169)] for _ in range(10)]
    fc_b = [0.0]*10

    for ep in range(epochs):
        total_loss = 0
        for img, label in zip(train_images, train_labels):
            loss = train_one(img, label, kernel, fc_w, fc_b, lr)
            total_loss += loss
        print(f"Epoch {ep+1}/{epochs}  Loss={total_loss/len(train_images):.4f}")

    # Save trained weights
    save_conv_kernel(kernel, "conv_weights.txt")
    save_fc_weights(fc_w, "fc_weights.txt")
    save_fc_biases(fc_b, "fc_biases.txt")

    print("Weights saved.")

    # Test accuracy
    correct = 0
    for img, label in zip(test_images, test_labels):
        _, _, _, _, _, probs = cnn_forward(img, kernel, fc_w, fc_b)
        if argmax(probs) == label:
            correct += 1

    print(f"Test accuracy: {correct}/{len(test_images)}")


# ==========================================================
#                         ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    # Change these paths to your directories
    train_dir = "mnist_train"
    test_dir  = "mnist_test"

    train_cnn(train_dir, test_dir, epochs=1, lr=0.01)
