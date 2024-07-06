# GPU Debugging

## Get and build the test code

```
git clone https://github.com/sekelle/cornerstone-octree cornerstone-octree.git

uenv start prgenv-gnu/24.2

CC=mpicc CXX=mpicxx \
cmake \
-DCMAKE_CUDA_ARCHITECTURES=90 \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_FLAGS_DEBUG=-g \
-DCMAKE_CUDA_FLAGS_DEBUG=-G \
-DCMAKE_CXX_FLAGS=-D__CUDA_ARCH_LIST__=900 \
-DCMAKE_CUDA_FLAGS=-ccbin=mpicxx \
-B build-cstone-g \
-S cornerstone-octree.git/

cmake --build build-cstone-g -t domain_gpu -j
```

More details about the `__CUDA_ARCH_LIST__` compiler flag in [issue#28](https://github.com/sekelle/cornerstone-octree/issues/28).

## Launch and Debug the test code

```
srun -N32 -n128 -t10 \
    ./cuda_visible_devices.sh \
    ./ddt-client \
    ./build.cstone-g/test/integration_mpi/domain_gpu
```
