import matplotlib.pyplot as plt
import time
from math import sqrt
import numpy as np

# Method 1: Recursive approach
def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# Method 2: Iterative approach
def fibonacci_iterative(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

# Method 3: Using Dynamic Programming (Memoization)
def fibonacci_memoization(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memoization(n-1, memo) + fibonacci_memoization(n-2, memo)
    return memo[n]

# Method 4: Using Dynamic Programming (Tabulation)
def fibonacci_tabulation(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    fib_table = [0] * (n + 1)
    fib_table[1] = 1
    for i in range(2, n + 1):
        fib_table[i] = fib_table[i-1] + fib_table[i-2]
    return fib_table[n]

# Method 5: Using Matrix Exponentiation
def fibonacci_matrix(n):
    def multiply_matrices(F, M):
        x = F[0][0] * M[0][0] + F[0][1] * M[1][0]
        y = F[0][0] * M[0][1] + F[0][1] * M[1][1]
        z = F[1][0] * M[0][0] + F[1][1] * M[1][0]
        w = F[1][0] * M[0][1] + F[1][1] * M[1][1]
        F[0][0], F[0][1], F[1][0], F[1][1] = x, y, z, w

    def power(F, n):
        if n == 0 or n == 1:
            return
        M = [[1, 1], [1, 0]]
        power(F, n // 2)
        multiply_matrices(F, F)
        if n % 2 != 0:
            multiply_matrices(F, M)

    F = [[1, 1], [1, 0]]
    if n == 0:
        return 0
    power(F, n - 1)
    return F[0][0]

# Method 6: Using Binet's Formula
def fibonacci_binet(n):
    phi = (1 + sqrt(5)) / 2
    return int(round((phi ** n - (-phi) ** -n) / sqrt(5)))


# Measure each method
nr = 30
n_values = range(nr)
methods = {
    'Recursive': fibonacci_recursive,
    'Iterative': fibonacci_iterative,
    'Memoization': fibonacci_memoization,
    'Tabulation': fibonacci_tabulation,
    'Matrix Exponentiation': fibonacci_matrix,
    'Binet\'s Formula': fibonacci_binet
}

# Create individual plots for each method
for method, function in methods.items():
    times = []
    for n in n_values:
        start_time = time.perf_counter()
        result = function(n)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    sum_time = np.sum(times)

    # Print method values and times
    print(f"\nMethod: {method}")
    print("Computation time:", np.round(sum_time, 10))
    print("Result:", np.round(function(nr), 10))


    plt.figure(figsize=(10, 5))  # New figure for each method
    plt.plot(n_values, times, marker='o', linestyle='-', color='b')
    plt.title(f'Computation Time for {method} Method')
    plt.xlabel('n')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')  # Log scale for clearer comparison
    plt.grid(True)
    plt.show()

