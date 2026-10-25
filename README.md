
# GPU-Based K-Means Clustering Using CUDA
### PGD AI — IIT Jodhpur | Problem 5: GPU Programming

A GPU-accelerated implementation of K-Means clustering using NVIDIA CUDA. This project parallelizes point-to-centroid distance computation and compares two centroid-update strategies: global atomic operations and block-privatised hierarchical reduction.

## Project Structure

```text
.
├── notebook/
│   └── KMeans_CUDA_Colab_Lab.ipynb
├── benchmark.py
├── kmeans_cpu.c
├── kmeans_cuda.cu
├── kmeans_io.py
├── kmeans_numpy.py
├── plot_results.py
├── verify.py
├── results/
├── CSL7520_Assignment_Problem5_KMeans_Report.docx
├── LICENSE
├── README.md
└── .gitignore
```

## Requirements

- NVIDIA GPU with CUDA support, or Google Colab GPU runtime.
- NVIDIA CUDA Toolkit and compatible drivers.
- GNU Make and a compatible C/C++ compiler.
- Python 3, NumPy, and Matplotlib.

## Project Information

- **Program:** PGD AI
- **Institution:** IIT Jodhpur
- **Problem:** Problem 5 — GPU Programming
- **Topic:** GPU-Based K-Means Clustering Using CUDA

