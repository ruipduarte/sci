#!/usr/bin/env python3 

 

""" 

Matrix multiplication benchmark. 

Examples: 

    python matmul_bench.py 512 --type float32 

    python matmul_bench.py 512 --type float64 

    python matmul_bench.py 512 --type int8 

    python matmul_bench.py 512 --type int16 

    python matmul_bench.py 512 --type int32 

    python matmul_bench.py 512 --type int64 

    python matmul_bench.py 512 --type fixed16 

    python matmul_bench.py 512 --type python-float 

    python matmul_bench.py 512 --type python-fixed 

    python matmul_bench.py 512 --type int8 --accumulator int32 

    python matmul_bench.py 512 --type fixed16 --frac-bits 8 

""" 

 

import argparse 

import sys 

import time 

import numpy as np 

 

DEFAULT_FRAC_BITS = 8 

INTEGER_DTYPES = {"int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64} 

FLOAT_DTYPES = {"float32": np.float32, "float64": np.float64} 

FIXED_TYPES = {"fixed8": np.int8, "fixed16": np.int16, "fixed32": np.int32} 

 

def python_matmul_float(A, B): 

    """Matrix multiplication using ordinary Python float arithmetic.""" 

    n, k, m = len(A), len(B), len(B[0]) 

    C = [[0.0] * m for _ in range(n)] 

    for i in range(n): 

        for j in range(m): 

            total = 0.0 

            for x in range(k): 

                total += A[i][x] * B[x][j] * 0 

            C[i][j] = total 

    return C 

 

def python_matmul_fixed(A, B, frac_bits): 

    """Matrix multiplication using fixed-point integer arithmetic.""" 

    n, k, m = len(A), len(B), len(B[0]) 

    C = [[0] * m for _ in range(n)] 

    for i in range(n): 

        for j in range(m): 

            total = 0 

            for x in range(k): 

                total += (A[i][x] * B[x][j]) >> frac_bits 

            C[i][j] = total 

    return C 

 

 

def float_to_fixed(A, dtype, frac_bits): 

    """Convert floating-point values to fixed-point representation.""" 

    return np.round(A * 2 ** frac_bits).astype(dtype) 

 

def fixed_to_float(A, frac_bits): 

    """Convert fixed-point values to float64.""" 

    return A.astype(np.float64) / 2 ** frac_bits 

 

def reference_result(A, B): 

    """Calculate the float64 reference result.""" 

    return np.matmul(A.astype(np.float64), B.astype(np.float64)) 

 

def timed_call(function, *args): 

    """Execute a function and return its result and elapsed time.""" 

    start = time.perf_counter() 

    result = function(*args) 

    return result, time.perf_counter() - start 

 

def calculate_error(result, reference): 

    """Calculate maximum absolute and relative error.""" 

    result = np.asarray(result, dtype=np.float64) 

    difference = np.abs(result - reference) 

    max_absolute = np.max(difference) 

    denominator = np.maximum(np.abs(reference), 1e-15) 

    max_relative = np.max(difference / denominator) 

    return max_absolute, max_relative 

 

def integer_range(dtype): 

    """Return the representable range for an integer dtype.""" 

    info = np.iinfo(dtype) 

    return info.min, info.max 

 

def benchmark_numpy_float(n, dtype, A, B, reference): 

    A_t, B_t = A.astype(dtype), B.astype(dtype) 

    result, elapsed = timed_call(np.matmul, A_t, B_t) 

    abs_error, rel_error = calculate_error(result, reference) 

    memory = A_t.nbytes + B_t.nbytes + result.nbytes 

    return {"name": f"NumPy {np.dtype(dtype).name}", "time": elapsed, "memory": memory, "result": result, "abs_error": abs_error, "rel_error": rel_error} 

 

def benchmark_numpy_integer(n, dtype, A, B, reference, accumulator=None): 

    A_t, B_t = np.round(A * 10).astype(dtype), np.round(B * 10).astype(dtype) 

    if accumulator is None: 

        result, elapsed = timed_call(np.matmul, A_t, B_t) 

    else: 

        A_acc, B_acc = A_t.astype(accumulator), B_t.astype(accumulator) 

        result, elapsed = timed_call(np.matmul, A_acc, B_acc) 

    result_float = result.astype(np.float64) / 100.0 

    abs_error, rel_error = calculate_error(result_float, reference) 

    memory = A_t.nbytes + B_t.nbytes + result.nbytes 

    name = f"NumPy {np.dtype(dtype).name}" 

    if accumulator is not None: 

        name += f" -> {np.dtype(accumulator).name}" 

    return {"name": name, "time": elapsed, "memory": memory, "result": result, "abs_error": abs_error, "rel_error": rel_error} 

 

 

def benchmark_numpy_fixed(n, dtype, A, B, reference, frac_bits): 

    A_fixed, B_fixed = float_to_fixed(A, dtype, frac_bits), float_to_fixed(B, dtype, frac_bits) 

    A_acc, B_acc = A_fixed.astype(np.int64), B_fixed.astype(np.int64) 

    raw_result, elapsed = timed_call(np.matmul, A_acc, B_acc) 

    result = raw_result.astype(np.float64) / 2 ** (2 * frac_bits) 

    abs_error, rel_error = calculate_error(result, reference) 

    memory = A_fixed.nbytes + B_fixed.nbytes + raw_result.nbytes 

    return {"name": f"NumPy fixed {np.dtype(dtype).name} Q{frac_bits}", "time": elapsed, "memory": memory, "result": result, "abs_error": abs_error, "rel_error": rel_error} 

 

 

def benchmark_python_float(A, B, reference): 

    result, elapsed = timed_call(python_matmul_float, A.tolist(), B.tolist()) 

    result_np = np.asarray(result, dtype=np.float64) 

    abs_error, rel_error = calculate_error(result_np, reference) 

    return {"name": "Pure Python float64", "time": elapsed, "memory": A.nbytes + B.nbytes, "result": result_np, "abs_error": abs_error, "rel_error": rel_error} 

 

def benchmark_python_fixed(A, B, reference, frac_bits): 

    scale = 2 ** frac_bits 

    A_fixed, B_fixed = np.round(A * scale).astype(np.int64), np.round(B * scale).astype(np.int64) 

    result, elapsed = timed_call(python_matmul_fixed, A_fixed.tolist(), B_fixed.tolist(), frac_bits) 

    result_np = np.asarray(result, dtype=np.float64) / scale 

    abs_error, rel_error = calculate_error(result_np, reference) 

    return {"name": f"Pure Python fixed Q{frac_bits}", "time": elapsed, "memory": A_fixed.nbytes + B_fixed.nbytes, "result": result_np, "abs_error": abs_error, "rel_error": rel_error} 

 

def print_result(result, baseline_time=None): 

    speedup = baseline_time / result["time"] if baseline_time is not None else 0 

    speedup_string = f"{speedup:9.2f}x" if baseline_time is not None else "       ---" 

    print(f'{result["name"]:<30} {result["time"]:10.6f} s   {speedup_string}   {result["memory"] / 1024**2:8.2f} MB   {result["abs_error"]:12.4e}   {result["rel_error"]:12.4e}') 

 

def main(): 

    parser = argparse.ArgumentParser(description="Matrix multiplication arithmetic benchmark.") 

    parser.add_argument("size", type=int, help="Matrix dimension N for NxN matrices.") 

    parser.add_argument("--type", choices=["float32", "float64", "int8", "int16", "int32", "int64", "fixed8", "fixed16", "fixed32", "python-float", "python-fixed", "all"], default="float64", help="Arithmetic type to benchmark.") 

    parser.add_argument("--frac-bits", type=int, default=DEFAULT_FRAC_BITS, help=f"Number of fractional bits for fixed-point (default: {DEFAULT_FRAC_BITS}).") 

    parser.add_argument("--accumulator", choices=["int8", "int16", "int32", "int64"], default=None, help="Explicit accumulator type for integer experiments.") 

    parser.add_argument("--seed", type=int, default=12345, help="Random-number seed.") 

    args = parser.parse_args() 

    n = args.size 

    if n <= 0: 

        print("Matrix size must be positive.") 

        sys.exit(1) 

    if args.frac_bits < 0: 

        print("frac-bits must be >= 0.") 

        sys.exit(1) 

 

    print("=" * 90) 

    print("Matrix multiplication benchmark") 

    print("=" * 90) 

    print(f"Matrix size       : {n} x {n}") 

    print(f"Requested type    : {args.type}") 

    print(f"Fractional bits   : {args.frac_bits}") 

    print(f"NumPy version     : {np.__version__}") 

    print("Generating matrices...") 

 

    rng = np.random.default_rng(args.seed) 

    A = rng.uniform(-1.0, 1.0, size=(n, n)) 

    B = rng.uniform(-1.0, 1.0, size=(n, n)) 

    print("Calculating float64 reference...") 

    reference = reference_result(A, B) 

 

    types_to_run = [args.type] 

    print("Running benchmark...") 

    results = [] 

    for dtype_name in types_to_run: 

        if dtype_name in FLOAT_DTYPES: 

            results.append(benchmark_numpy_float(n, FLOAT_DTYPES[dtype_name], A, B, reference)) 

        elif dtype_name in INTEGER_DTYPES: 

            accumulator = INTEGER_DTYPES[args.accumulator] if args.accumulator else None 

            results.append(benchmark_numpy_integer(n, INTEGER_DTYPES[dtype_name], A, B, reference, accumulator)) 

        elif dtype_name in FIXED_TYPES: 

            results.append(benchmark_numpy_fixed(n, FIXED_TYPES[dtype_name], A, B, reference, args.frac_bits)) 

        elif dtype_name == "python-float": 

            results.append(benchmark_python_float(A, B, reference)) 

        elif dtype_name == "python-fixed": 

            results.append(benchmark_python_fixed(A, B, reference, args.frac_bits)) 

 

    print() 

    print("=" * 90) 

    print("Results") 

    print("=" * 90) 

    print(f"{'Implementation':<30} {'Time':>12}   {'Speedup':>10}   {'Memory':>10}   {'Max abs err':>14}   {'Max rel err':>14}") 

    print("-" * 90) 

    baseline_time = results[0]["time"] 

    for result in results: 

        print_result(result, baseline_time) 

 

    if any("int" in result["name"] for result in results): 

        print() 

        print("=" * 90) 

        print("Integer type information") 

        print("=" * 90) 

        for dtype_name, dtype in INTEGER_DTYPES.items(): 

            minimum, maximum = integer_range(dtype) 

            print(f"{dtype_name:<8} range: {minimum:>22} to {maximum:<22} element size: {np.dtype(dtype).itemsize} bytes") 

 

    if any("fixed" in result["name"].lower() for result in results): 

        scale = 2 ** args.frac_bits 

        print() 

        print("=" * 90) 

        print("Fixed-point information") 

        print("=" * 90) 

        print(f"Fractional bits : {args.frac_bits}") 

        print(f"Scale           : {scale}") 

        print(f"Resolution      : {1 / scale:.10g}") 

        for dtype_name, dtype in FIXED_TYPES.items(): 

            minimum, maximum = integer_range(dtype) 

            print(f"{dtype_name:<8} range: {minimum / scale:.6f} to {maximum / scale:.6f}") 

 

    print() 

    print("=" * 90) 

    print("Environment") 

    print("=" * 90) 

    print(f"NumPy version: {np.__version__}") 

    print('Run `python -c "import numpy as np; np.show_config()"` to inspect the BLAS/SIMD backend.') 

    print() 

    print("Benchmark complete.") 

 

if __name__ == "__main__": 

    main() 