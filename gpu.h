#include <chrono>
#include "lcg.h"
#include "util.h"

uint64_t millis() {return (std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch())).count();}

#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
	fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
	exit(code);
  }
}

#define SETGPU(gpuId) cudaSetDevice(GPU_ID);\
	GPU_ASSERT(cudaPeekAtLastError());\
	GPU_ASSERT(cudaDeviceSynchronize());\
	GPU_ASSERT(cudaPeekAtLastError());
#define GPUMALLOC(buf,count)GPU_ASSERT(cudaMallocManaged(&buf, sizeof(*buf) * count));\
	GPU_ASSERT(cudaPeekAtLastError()); 
	
	
namespace device_intrinsics { //region DEVICE INTRINSICS
    #define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__

    #if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
    #define PXL_GLOBAL_PTR   "l"
    #else
    #define PXL_GLOBAL_PTR   "r"
    #endif

    DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_local_l1(const void* const ptr)
    {
      asm("prefetch.local.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
    }

    DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_uniform(const void* const ptr)
    {
      asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
    }

    DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_local_l2(const void* const ptr)
    {
      asm("prefetch.local.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
    }

    #if __CUDA__ < 10
    #define __ldg(ptr) (*(ptr))
    #endif
}
using namespace device_intrinsics;