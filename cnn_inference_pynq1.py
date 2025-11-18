
import os
import math


# ==========================================================
#       PGM (P5) READER — Simple + Fast (no numpy)
# ==========================================================
def load_pgm_p5(filename):
    """
    Load a 28x28 P5 PGM image and return a normalized 28x28 matrix.
    """
    with open(filename, "rb") as f:
        assert f.readline().strip() == b'P5'          # Magic number
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()                       # Skip comments

        w, h = map(int, line.split())
        maxval = int(f.readline())
        assert w == 28 and h == 28

        data = f.read(28 * 28)
        img = []
        idx = 0
        for i in range(28):
            row = []
            for j in range(28):
                row.append(data[idx] / maxval)
                idx += 1
            img.append(row)
        return img


# ==========================================================
#                    WEIGHT LOADERS
# ==========================================================
def load_conv_kernel(filename):
    kernel = []
    with open(filename, "r") as f:
        for line in f:
            kernel.append([float(x) for x in line.strip().split(",")])
    return kernel


def load_fc_weights(filename):
    weights = []
    with open(filename, "r") as f:
        for line in f:
            weights.append([float(x) for x in line.strip().split(",")])
    return weights


def load_fc_biases(filename):
    biases = []
    with open(filename, "r") as f:
        for line in f:
            biases.append(float(line.strip()))
    return biases


# ==========================================================
#            CNN: Forward-Only (Optimized)
# ==========================================================
def conv2d(image, kernel):
    out = [[0]*26 for _ in range(26)]
    for i in range(26):
        row_i = image[i]
        row_i1 = image[i+1]
        row_i2 = image[i+2]
        for j in range(26):
            out[i][j] = (
                row_i[j]   * kernel[0][0] +
                row_i[j+1] * kernel[0][1] +
                row_i[j+2] * kernel[0][2] +
                row_i1[j]   * kernel[1][0] +
                row_i1[j+1] * kernel[1][1] +
                row_i1[j+2] * kernel[1][2] +
                row_i2[j]   * kernel[2][0] +
                row_i2[j+1] * kernel[2][1] +
                row_i2[j+2] * kernel[2][2]
            )
    return out


def relu(fm):
    return [[(v if v > 0 else 0) for v in row] for row in fm]


def maxpool2x2(fm):
    out = []
    for i in range(0, 26, 2):
        row = []
        fm_row = fm[i]
        fm_row1 = fm[i+1]
        for j in range(0, 26, 2):
            m = fm_row[j]
            v2 = fm_row[j+1]
            if v2 > m: m = v2
            v3 = fm_row1[j]
            if v3 > m: m = v3
            v4 = fm_row1[j+1]
            if v4 > m: m = v4
            row.append(m)
        out.append(row)
    return out


def flatten(fm):
    flat = []
    for row in fm:
        flat.extend(row)
    return flat


def fc_layer(vec, weights, biases):
    out = []
    for i in range(len(weights)):
        s = biases[i]
        w_row = weights[i]
        for j in range(len(w_row)):
            s += w_row[j] * vec[j]
        out.append(s)
    return out


def softmax(logits):
    exps = [math.exp(x) for x in logits]
    s = sum(exps)
    return [x/s for x in exps]


# ==========================================================
#               COMPLETE FORWARD INFERENCE
# ==========================================================
def cnn_inference(image, conv_k, fc_w, fc_b):
    conv = conv2d(image, conv_k)
    rel  = relu(conv)
    pool = maxpool2x2(rel)
    flat = flatten(pool)
    logits = fc_layer(flat, fc_w, fc_b)
    probs = softmax(logits)
    return probs


# ==========================================================
#               DIRECTORY-LEVEL INFERENCE
# ==========================================================
def infer_directory(image_dir, conv_k, fc_w, fc_b):
    correct = 0
    total = 0

    for fname in os.listdir(image_dir):
        if not fname.endswith(".pgm"):
            continue

        label = int(fname.split("_")[0])  # filename encodes label
        img = load_pgm_p5(os.path.join(image_dir, fname))

        probs = cnn_inference(img, conv_k, fc_w, fc_b)
        pred = max(range(10), key=lambda i: probs[i])

        total += 1
        if pred == label:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy on {image_dir}: {correct}/{total} = {accuracy:.4f}")
    return accuracy


# ==========================================================
#                      ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    # Load trained weights (must be copied to the Pynq board)
    conv_k = load_conv_kernel("conv_weights.txt")
    fc_w   = load_fc_weights("fc_weights.txt")
    fc_b   = load_fc_biases("fc_biases.txt")

    # Directory containing PGM test images
    test_dir = "mnist_test"

    infer_directory(test_dir, conv_k, fc_w, fc_b)
