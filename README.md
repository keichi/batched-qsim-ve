# batched-qsim-ve

## Building

NEC Vector Engine

```
cmake -DCMAKE_CXX_FLAGS="-std=c++17" \
      -DCMAKE_TOOLCHAIN_FILE=/opt/nec/ve/share/cmake/toolchainVE.cmake \
      -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
```

Apple Silicon

```
cmake -DCMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include -Xpreprocessor -fopenmp" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix libomp)/lib -lomp" \
      -DCMAKE_MODULE_LINKER_FLAGS="-L$(brew --prefix)/lib -lomp" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
```

Intel CPU

```
cmake -DCMAKE_CXX_COMPILER=icpc \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-axCORE-AVX512" \
      ..
```
