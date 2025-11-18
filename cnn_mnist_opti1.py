import os
import math

# ==========================================================
#          FIXED-POINT HELPERS (Q7 integer arithmetic)
# ==========================================================
Q7_SHIFT = 7      # scale = 2^7 = 128
Q7_SCALE = 1 << Q7_SHIFT


def to_q7(x):
    """Convert float to Q7 int8."""
    v = int(x * Q7_SCALE)
    if v < -128: v = -128
    if v > 127: v = 127
    return v


def softmax(logits):
    """Logits are still float for simplicity."""
    exps = [math.exp(l) for l in logits]
    s = sum(exps)
    return [e/s for e in exps]


# ==========================================================
#       PGM READER (returns int8 Q7 pixel matrix)
# ==========================================================
def load_pgm_p5(filename):
    with open(filename, "rb") as f:
        assert f.readline().strip() == b'P5'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        assert w == 28 and h == 28

        data = f.read(28 * 28)

        img = []
        idx = 0
        for i in range(28):
            row = []
            for j in range(28):
                # Normalize and convert to Q7
                row.append(to_q7(data[idx] / maxval))
                idx += 1
            img.append(row)
        return img


# ==========================================================
#                 LOAD INT8 / INT16 WEIGHTS
# ==========================================================
def load_conv_kernel_int8(filename):
    kernel = []
    with open(filename, "r") as f:
        for line in f:
            row = [int(x) for x in line.strip().split(",")]
            kernel.append(row)
    return kernel  # values expected in int8


def load_fc_weights_int8(filename):
    weights = []
    with open(filename, "r") as f:
        for line in f:
            weights.append([int(x) for x in line.strip().split(",")])
    return weights


def load_fc_biases_int16(filename):
    biases = []
    with open(filename, "r") as f:
        for line in f:
            biases.append(int(line.strip()))
    return biases


# ==========================================================
#               FIXED-POINT CONVOLUTION (FAST)
# ==========================================================
def conv2d_q7(image, kernel):
    """
    image  : 28x28 int8 (Q7)
    kernel : 3x3 int8 (Q7)
    output : 26x26 int16 (Q14)
    """
    out = [[0]*26 for _ in range(26)]

    k00, k01, k02 = kernel[0]
    k10, k11, k12 = kernel[1]
    k20, k21, k22 = kernel[2]

    for i in range(26):
        row_i  = image[i]
        row_i1 = image[i+1]
        row_i2 = image[i+2]

        oi = out[i]
        for j in range(26):
            # manual unrolling, accumulate in int16
            s = 0
            s += row_i[j]   * k00
            s += row_i[j+1] * k01
            s += row_i[j+2] * k02

            s += row_i1[j]   * k10
            s += row_i1[j+1] * k11
            s += row_i1[j+2] * k12

            s += row_i2[j]   * k20
            s += row_i2[j+1] * k21
            s += row_i2[j+2] * k22

            oi[j] = s  # Q14
    return out


# ==========================================================
#                 FIXED-POINT RELU + POOLING
# ==========================================================
def relu_q14(fm):
    return [[(v if v > 0 else 0) for v in row] for row in fm]


def maxpool2x2_q14(fm):
    out = []
    for i in range(0, 26, 2):
        row = []
        fm0 = fm[i]
        fm1 = fm[i+1]
        for j in range(0, 26, 2):
            m = fm0[j]
            v = fm0[j+1]
            if v > m: m = v
            v = fm1[j]
            if v > m: m = v
            v = fm1[j+1]
            if v > m: m = v
            row.append(m)
        out.append(row)
    return out


# ==========================================================
#              FC Layer (fixed-point inputs)
# ==========================================================
def flatten_q14(fm):
    out = []
    for row in fm:
        out.extend(row)
    return out


def fc_layer_q14(vec, weights, biases):
    """
    vec: Q14 int16
    weight: Q7 int8
    output: float (converted at end)
    """
    out = []
    for i in range(len(weights)):
        s = biases[i]  # bias stored as int16 (approx Q14)
        w_row = weights[i]
        for j in range(len(w_row)):
            # Multiply Q14 * Q7 = Q21 → shift back to Q14
            s += (vec[j] * w_row[j]) >> Q7_SHIFT
        out.append(s / float(1 << Q7_SHIFT))  # convert to float for softmax
    return out


# ==========================================================
#               COMPLETE FIXED-POINT FORWARD PASS
# ==========================================================
def cnn_inference_q7(image, conv_k, fc_w, fc_b):
    conv = conv2d_q7(image, conv_k)
    rel  = relu_q14(conv)
    pool = maxpool2x2_q14(rel)
    flat = flatten_q14(pool)
    logits = fc_layer_q14(flat, fc_w, fc_b)
    return softmax(logits)


# ==========================================================
#                   DIRECTORY INFERENCE
# ==========================================================
def infer_directory_q7(image_dir, conv_k, fc_w, fc_b):
    correct = 0
    total = 0

    for fname in os.listdir(image_dir):
        if not fname.endswith(".pgm"):
            continue

        label = int(fname.split("_")[0])
        img = load_pgm_p5(os.path.join(image_dir, fname))

        probs = cnn_inference_q7(img, conv_k, fc_w, fc_b)
        pred = max(range(10), key=lambda i: probs[i])

        total += 1
        if pred == label:
            correct += 1

    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    return correct, total

# ==========================================================
#                     ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    conv_k = load_conv_kernel_int8("conv_weights_int8.txt")
    fc_w   = load_fc_weights_int8("fc_weights_int8.txt")
    fc_b   = load_fc_biases_int16("fc_biases_int16.txt")

    infer_directory_q7("mnist_test", conv_k, fc_w, fc_b)
