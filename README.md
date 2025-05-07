# veqsim

```
python3 -m install nanobind veda
```

## Building

### NEC Vector Engine

```
cmake -DCMAKE_CXX_FLAGS="-std=c++17" \
      -DCMAKE_TOOLCHAIN_FILE=/opt/nec/ve/share/cmake/toolchainVE.cmake \
      -DCMAKE_EXE_LINKER_FLAGS="-fopenmp -Wl,'-z muldefs'" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
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
cmake -DCMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include -Xpreprocessor -fopenmp" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix libomp)/lib -lomp" \
      -DCMAKE_MODULE_LINKER_FLAGS="-L$(brew --prefix)/lib -lomp" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
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
