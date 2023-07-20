# batched-qsim-ve

## Building (Vector Engine)

```
cmake -DCMAKE_CXX_FLAGS="-std=c++17" \
      -DCMAKE_TOOLCHAIN_FILE=/opt/nec/ve/share/cmake/toolchainVE.cmake \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
```

## Building (Intel CPU)

```
cmake -DCMAKE_CXX_COMPILER=icpc \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-axCORE-AVX512" \
      ..
```
