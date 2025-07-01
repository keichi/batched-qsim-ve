# veqsim

## Requirements

- GCC 8+
- Python 3.8+
- CMake 3.18+
- Recent nanobind and VEDA (installed via pip)

## Building

### NEC Vector Engine

```
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
```

### NVIDIA A100

Make sure cuStateVec v1.4.0 or later is installed.

```
cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
```

### Apple Silicon

Make sure libomp is installed.

```
cmake -DOpenMP_ROOT=$(brew --prefix)/opt/libomp \
      ..
```

### Intel CPU

```
cmake -DCMAKE_CXX_COMPILER=icpc \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-axCORE-AVX512" \
      ..
```

## Citing

Please cite the following paper if you use veqsim:

- Keichi Takahashi, Toshio Mori, Hiroyuki Takizawa, "Prototype of a Batched Quantum Circuit
  Simulator for the Vector Engine," 4th International Workshop on Quantum Computing Software held in
  conjunction with SC23: The International Conference for High Performance Computing, Networking,
  Storage, and Analysis, Nov. 2023. [10.1145/3624062.3624226](https://doi.org/10.1145/3624062.3624226)
