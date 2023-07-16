## batched-qsim-ve

```
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS_RELEASE="-O4 -finline-functions -report-all -finline-max-depth=3" \
      -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
      -DCMAKE_TOOLCHAIN_FILE=/opt/nec/ve/share/cmake/toolchainVE.cmake \
      ..

```
