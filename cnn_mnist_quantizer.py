# ==========================================================
#       FIXED-POINT QUANTIZATION HELPER FUNCTIONS
# ==========================================================

Q7_SHIFT = 7          # Q7: scale factor = 2^7 = 128
Q7_SCALE = 1 << Q7_SHIFT
Q14_SHIFT = 14        # Q14 for biases
Q14_SCALE = 1 << Q14_SHIFT


def float_to_q7(x):
    """Convert a float to Q7 int8."""
    v = int(round(x * Q7_SCALE))
    if v < -128: v = -128
    if v > 127: v = 127
    return v


def float_to_q14(x):
    """Convert a float to Q14 int16."""
    v = int(round(x * Q14_SCALE))
    if v < -32768: v = -32768
    if v > 32767: v = 32767
    return v


# ==========================================================
#                LOAD FLOAT WEIGHTS (CSV)
# ==========================================================

def load_float_matrix(filename):
    mat = []
    with open(filename, "r") as f:
        for line in f:
            row = [float(x) for x in line.strip().split(",")]
            mat.append(row)
    return mat


def load_float_vector(filename):
    vec = []
    with open(filename, "r") as f:
        for line in f:
            vec.append(float(line.strip()))
    return vec


# ==========================================================
#            SAVE INTEGER WEIGHTS (CSV output)
# ==========================================================

def save_int_matrix(mat, filename):
    with open(filename, "w") as f:
        for row in mat:
            f.write(",".join(str(int(x)) for x in row) + "\n")


def save_int_vector(vec, filename):
    with open(filename, "w") as f:
        for v in vec:
            f.write(str(int(v)) + "\n")


# ==========================================================
#         MAIN QUANTIZATION PIPELINE (FLOAT → INT)
# ==========================================================

def quantize_weights(conv_w_file, fc_w_file, fc_b_file):
    # Load FP32 weights
    conv_f = load_float_matrix(conv_w_file)
    fc_w_f = load_float_matrix(fc_w_file)
    fc_b_f = load_float_vector(fc_b_file)

    # Quantize convolution kernel to Q7 int8
    conv_q7 = [[float_to_q7(v) for v in row] for row in conv_f]

    # Quantize FC weights to Q7 int8
    fc_w_q7 = []
    for row in fc_w_f:
        fc_w_q7.append([float_to_q7(v) for v in row])

    # Quantize FC biases to Q14 int16
    fc_b_q14 = [float_to_q14(b) for b in fc_b_f]

    # Write output
    save_int_matrix(conv_q7, "conv_weights_int8.txt")
    save_int_matrix(fc_w_q7, "fc_weights_int8.txt")
    save_int_vector(fc_b_q14, "fc_biases_int16.txt")

    print("Quantization complete.")
    print("Generated files:")
    print("  conv_weights_int8.txt")
    print("  fc_weights_int8.txt")
    print("  fc_biases_int16.txt")


# ==========================================================
#                          ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    quantize_weights(
        "conv_weights.txt",
        "fc_weights.txt",
        "fc_biases.txt"
    )
