import matplotlib.pyplot as plt
import numpy as np
import time


# Implement sorting algorithms

# Quick Sort
def quick_sort(arr, start, end, iterations):
    if start < end:
        pi, new_iterations = partition(arr, start, end)
        iterations[0] += new_iterations
        quick_sort(arr, start, pi - 1, iterations)
        quick_sort(arr, pi + 1, end, iterations)


def partition(arr, start, end):
    pivot = arr[end]
    i = start - 1
    iterations = 0
    for j in range(start, end):
        iterations += 1
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[end] = arr[end], arr[i + 1]
    return i + 1, iterations


# Merge Sort
def merge_sort(arr, iterations):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L, iterations)
        merge_sort(R, iterations)

        i = j = k = 0

        while i < len(L) and j < len(R):
            iterations[0] += 1
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


# Heap Sort
def heapify(arr, n, i, iterations):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[l] > arr[largest]:
        largest = l

    if r < n and arr[r] > arr[largest]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest, iterations)


def heap_sort(arr, iterations):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, iterations)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        iterations[0] += 1
        heapify(arr, i, 0, iterations)


# Shell Sort
def shell_sort(arr, iterations):
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
                iterations[0] += 1
            arr[j] = temp
        gap //= 2


array_lengths = list(range(1, 1001, 25))
quick_sort_iterations = []
merge_sort_iterations = []
heap_sort_iterations = []
shell_sort_iterations = []

for length in array_lengths:
    arr = np.random.randint(1, 1000, length).tolist()

    iterations = [0]
    quick_sort(arr.copy(), 0, length - 1, iterations)
    quick_sort_iterations.append(iterations[0])
    print(f'Quick Sort - Array Length: {length}, Iterations: {iterations[0]}')

    iterations = [0]
    merge_sort(arr.copy(), iterations)
    merge_sort_iterations.append(iterations[0])
    print(f'Merge Sort - Array Length: {length}, Iterations: {iterations[0]}')

    iterations = [0]
    heap_sort(arr.copy(), iterations)
    heap_sort_iterations.append(iterations[0])
    print(f'Heap Sort - Array Length: {length}, Iterations: {iterations[0]}')

    iterations = [0]
    shell_sort(arr.copy(), iterations)  # Replacing radix sort with shell sort
    shell_sort_iterations.append(iterations[0])
    print(f'Shell Sort - Array Length: {length}, Iterations: {iterations[0]}')

    print('\n')

plt.figure(figsize=[12, 8])
plt.plot(array_lengths, quick_sort_iterations, label='Quick Sort', marker='o')
plt.plot(array_lengths, merge_sort_iterations, label='Merge Sort', marker='s')
plt.plot(array_lengths, heap_sort_iterations, label='Heap Sort', marker='^')
plt.plot(array_lengths, shell_sort_iterations, label='Shell Sort', marker='d')
plt.xlabel('Array Length')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations for Different Sorting Algorithms')
plt.show()

# Initialize lists to store execution times
quick_sort_times = []
merge_sort_times = []
heap_sort_times = []
shell_sort_times = []

for length in array_lengths:
    arr = np.random.randint(1, 1000, length).tolist()

    # Measure Quick Sort time
    start_time = time.time()
    iterations = [0]
    quick_sort(arr.copy(), 0, length - 1, iterations)
    end_time = time.time()
    quick_sort_times.append(end_time - start_time)

    # Measure Merge Sort time
    start_time = time.time()
    iterations = [0]
    merge_sort(arr.copy(), iterations)
    end_time = time.time()
    merge_sort_times.append(end_time - start_time)

    # Measure Heap Sort time
    start_time = time.time()
    iterations = [0]
    heap_sort(arr.copy(), iterations)
    end_time = time.time()
    heap_sort_times.append(end_time - start_time)

    # Measure Shell Sort time
    start_time = time.time()
    iterations = [0]
    shell_sort(arr.copy(), iterations)
    end_time = time.time()
    shell_sort_times.append(end_time - start_time)

# Plot for Quick Sort
plt.figure(figsize=[12, 8])
plt.plot(array_lengths, quick_sort_times, label='Quick Sort', marker='o')
plt.xlabel('Array Length')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time for Quick Sort')
plt.legend()
plt.show()

# Plot for Merge Sort
plt.figure(figsize=[12, 8])
plt.plot(array_lengths, merge_sort_times, label='Merge Sort', marker='s')
plt.xlabel('Array Length')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time for Merge Sort')
plt.legend()
plt.show()

# Plot for Heap Sort
plt.figure(figsize=[12, 8])
plt.plot(array_lengths, heap_sort_times, label='Heap Sort', marker='^')
plt.xlabel('Array Length')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time for Heap Sort')
plt.legend()
plt.show()

# Plot for Shell Sort
plt.figure(figsize=[12, 8])
plt.plot(array_lengths, shell_sort_times, label='Shell Sort', marker='d')
plt.xlabel('Array Length')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time for Shell Sort')
plt.legend()
plt.show()
