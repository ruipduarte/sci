import random
import time

# Use the HW-accelerated conv2d_q7 from the driver we just fixed
from conv2d_q7_axis_hw import conv2d_q7 as conv2d_q7_hw


def sat16(x):
    if x > 32767:
        return 32767
    elif x < -32768:
        return -32768
    return x


def conv2d_q7_sw(image, kernel):
    """
    Pure Python reference (Q7 → Q14, with int16 saturation)
    image  : 28x28 int (Q7)
    kernel : 3x3  int (Q7)
    output : 26x26 int (Q14)
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

            oi[j] = sat16(s)
    return out


def random_q7_image():
    return [[random.randint(-128, 127) for _ in range(28)] for _ in range(28)]


def random_q7_kernel():
    return [[random.randint(-128, 127) for _ in range(3)] for _ in range(3)]


def compare_feature_maps(sw, hw):
    mismatches = 0
    for i in range(26):
        for j in range(26):
            if sw[i][j] != hw[i][j]:
                mismatches += 1
    return mismatches


def run_self_test(num_tests=10, bitfile="conv2d_q7_axis.bit"):
    print(f"Running {num_tests} HW vs SW conv2d_q7 tests...")
    total_mismatches = 0

    # Warm-up (loads overlay)
    img0 = random_q7_image()
    ker0 = random_q7_kernel()
    _ = conv2d_q7_hw(img0, ker0, bitfile=bitfile)

    sw_time_total = 0.0
    hw_time_total = 0.0

    for t in range(num_tests):
        image = random_q7_image()
        kernel = random_q7_kernel()

        t0 = time.time()
        fm_sw = conv2d_q7_sw(image, kernel)
        sw_time = time.time() - t0

        t0 = time.time()
        fm_hw = conv2d_q7_hw(image, kernel, bitfile=bitfile)
        hw_time = time.time() - t0

        sw_time_total += sw_time
        hw_time_total += hw_time

        mismatches = compare_feature_maps(fm_sw, fm_hw)
        total_mismatches += mismatches

        print(f"Test {t+1}/{num_tests}: "
              f"SW {sw_time*1e3:.2f} ms, HW {hw_time*1e3:.2f} ms, "
              f"mismatches = {mismatches}")

    print("\n=== Summary ===")
    print(f"Total mismatching elements over all tests: {total_mismatches}")
    print(f"Average SW time per call: {sw_time_total/num_tests*1e3:.2f} ms")
    print(f"Average HW time per call: {hw_time_total/num_tests*1e3:.2f} ms")

    if total_mismatches == 0:
        print("All tests PASSED: HW matches SW exactly.")
    else:
        print("WARNING: mismatches detected between HW and SW results.")


# Actually run the test
run_self_test(num_tests=10, bitfile="conv2d_q7_axis.bit")
