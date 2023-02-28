crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/detail/CUDAHooks.h]

/// Callback to initialize THC Magma, which is
/// implemented in torch_cuda_cu
///
lazy_static!{
    /*
    function<void(void)> THCMagma_init;
    */
}

/// The real implementation of CUDAHooksInterface
///
pub struct CUDAHooks {
    base: CUDAHooksInterface,
}

impl CUDAHooks {
    
    pub fn new(x: CUDAHooksArgs) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn initcuda(&self) -> Box<THCState> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_device_from_ptr(&self, data: *mut void) -> Device {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_pinned_ptr(&self, data: *mut void) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_default_cuda_generator(&self, device_index: DeviceIndex) -> &Generator {
        let device_index: DeviceIndex = device_index.unwrap_or(-1);

        todo!();
        /*
        
        */
    }
    
    pub fn hascuda(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn hasmagma(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn nvrtc(&self) -> &NVRTC {
        
        todo!();
        /*
        
        */
    }
    
    pub fn current_device(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_primary_context(&self, device_index: i64) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_devce_index_with_primary_context(&self) -> Option<i64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_cuda_device_allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_pinned_memory_allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compiled_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn compiled_with_mio_pen(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn supports_dilated_convolution_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn supports_depthwise_convolution_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn hascudart(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn versioncudart(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn version_cu_dnn(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn show_config(&self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    pub fn batchnorm_min_epsilon_cu_dnn(&self) -> f64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn cu_fft_get_plan_cache_max_size(&self, device_index: i64) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn cu_fft_set_plan_cache_max_size(&self, 
        device_index: i64,
        max_size:     i64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn cu_fft_get_plan_cache_size(&self, device_index: i64) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn cu_fft_clear_plan_cache(&self, device_index: i64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_num_gpu_s(&self) -> i32 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn device_synchronize(&self, device_index: i64)  {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/detail/CUDAHooks.cpp]

impl CUDAHooks {
    
    /**
      | NB: deleter is dynamic, because we need it to
      | live in a separate compilation unit (alt is to
      | have another method in hooks, but let's not if
      | we don't need to!)
      |
      */
    pub fn initcuda(&self) -> Box<THCState> {
        
        todo!();
        /*
            C10_LOG_API_USAGE_ONCE("aten.init.cuda");
      THCState* thc_state = THCState_alloc();

      THCudaInit(thc_state);
      if (THCMagma_init)
        THCMagma_init();
      return unique_ptr<THCState, void (*)(THCState*)>(
          thc_state, [](THCState* p) {
            if (p)
              THCState_free(p);
          });
        */
    }
    
    pub fn get_default_cuda_generator(&self, device_index: DeviceIndex) -> &Generator {
        
        todo!();
        /*
            return cuda::detail::getDefaultCUDAGenerator(device_index);
        */
    }
    
    pub fn get_device_from_ptr(&self, data: *mut void) -> Device {
        
        todo!();
        /*
            return cuda::getDeviceFromPtr(data);
        */
    }
    
    pub fn is_pinned_ptr(&self, data: *mut void) -> bool {
        
        todo!();
        /*
            // First check if driver is broken/missing, in which case PyTorch CPU
      // functionalities should still work, we should report `false` here.
      if (!CUDAHooks::hasCUDA()) {
        return false;
      }
      // cudaPointerGetAttributes grabs context on the current device, so we set
      // device to one that already has context, if exists.
      OptionalDeviceGuard device_guard;
      auto primary_ctx_device_index = CUDAHooks::getDevceIndexWithPrimaryContext();
      if (primary_ctx_device_index.has_value()) {
        device_guard.reset_device(Device(DeviceType_CUDA, *primary_ctx_device_index));
      }
      cudaPointerAttributes attr;
      cudaError_t err = cudaPointerGetAttributes(&attr, data);
    #ifndef __HIP_PLATFORM_HCC__
      if (err == cudaErrorInvalidValue) {
        cudaGetLastError();
        return false;
      }
      AT_CUDA_CHECK(err);
    #else
      // HIP throws hipErrorUnknown here
      if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
      }
    #endif
    #if CUDA_VERSION >= 10000
      return attr.type == cudaMemoryTypeHost;
    #else
      return attr.memoryType == cudaMemoryTypeHost;
    #endif
        */
    }
    
    pub fn hascuda(&self) -> bool {
        
        todo!();
        /*
            return cuda::is_available();
        */
    }
    
    pub fn hasmagma(&self) -> bool {
        
        todo!();
        /*
            #ifdef USE_MAGMA
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn has_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return AT_CUDNN_ENABLED();
        */
    }
}

#[cfg(USE_DIRECT_NVRTC)]
pub fn load_nvrtc() -> Pair<Box<DynamicLibrary>,*mut NVRTC> {
    
    todo!();
        /*
            return make_pair(nullptr, load_nvrtc());
        */
}

#[cfg(not(USE_ROCM))]
pub fn load_nvrtc() -> Pair<Box<DynamicLibrary>,*mut NVRTC> {
    
    todo!();
        /*
            return make_pair(nullptr, &detail::lazyNVRTC);
        */
}

#[cfg(not(any(USE_DIRECT_NVRTC,not(USE_ROCM))))]
pub fn load_nvrtc() -> Pair<Box<DynamicLibrary>,*mut NVRTC> {
    
    todo!();
        /*
            #if defined(_WIN32)
      string libcaffe2_nvrtc = "caffe2_nvrtc.dll";
    #elif defined(__APPLE__)
      string libcaffe2_nvrtc = "libcaffe2_nvrtc.dylib";
    #else
      string libcaffe2_nvrtc = "libcaffe2_nvrtc.so";
    #endif
      unique_ptr<DynamicLibrary> libnvrtc_stub(
          new DynamicLibrary(libcaffe2_nvrtc.c_str()));
      auto fn = (NVRTC * (*)()) libnvrtc_stub->sym("load_nvrtc");
      return make_pair(move(libnvrtc_stub), fn());
        */
}

impl CUDAHooks {
    
    pub fn nvrtc(&self) -> &NVRTC {
        
        todo!();
        /*
            // must hold onto DynamicLibrary otherwise it will unload
      static auto handle = load_nvrtc();
      return *handle.second;
        */
    }
    
    pub fn current_device(&self) -> i64 {
        
        todo!();
        /*
            int device;
      cudaError_t err = cudaGetDevice(&device);
      if (err == cudaSuccess) {
        return device;
      }
      return -1;
        */
    }
    
    pub fn has_primary_context(&self, device_index: i64) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(device_index >= 0 && device_index < device_count(),
                  "hasPrimaryContext expects a valid device index, but got device_index=", device_index);
      unsigned int ctx_flags;
      // In standalone tests of cuDevicePrimaryCtxGetState, I've seen the "active" argument end up with weird
      // (garbage-looking nonzero) values when the context is not active, unless I initialize it to zero.
      int ctx_is_active = 0;
      AT_CUDA_DRIVER_CHECK(CUDAHooks::nvrtc().cuDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
      return ctx_is_active == 1;
        */
    }
    
    pub fn get_devce_index_with_primary_context(&self) -> Option<i64> {
        
        todo!();
        /*
            // check current device first
      i64 current_device_index = CUDAHooks::current_device();
      if (current_device_index >= 0) {
        if (CUDAHooks::hasPrimaryContext(current_device_index)) {
          return current_device_index;
        }
      }
      for (i64 device_index = 0; device_index < CUDAHooks::getNumGPUs(); device_index++) {
        if (device_index == current_device_index) continue;
        if (CUDAHooks::hasPrimaryContext(device_index)) {
          return device_index;
        }
      }
      return nullopt;
        */
    }
    
    pub fn get_pinned_memory_allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
            return getPinnedMemoryAllocator();
        */
    }
    
    pub fn get_cuda_device_allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
            return getCUDADeviceAllocator();
        */
    }
    
    pub fn compiled_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return AT_CUDNN_ENABLED();
        */
    }
    
    pub fn compiled_with_mio_pen(&self) -> bool {
        
        todo!();
        /*
            return AT_ROCM_ENABLED();
        */
    }
    
    pub fn supports_dilated_convolution_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            #if AT_CUDNN_ENABLED()
      // NOTE: extra parenthesis around numbers disable clang warnings about
      // dead code
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn supports_depthwise_convolution_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            #if AT_CUDNN_ENABLED()
      cudaDeviceProp* prop = getCurrentDeviceProperties();
      // Check for Volta cores
      if (prop->major >= 7) {
        return true;
      } else {
        return false;
      }
    #else
      return false;
    #endif
        */
    }
    
    pub fn version_cu_dnn(&self) -> i64 {
        
        todo!();
        /*
            #if AT_CUDNN_ENABLED()
      return CUDNN_VERSION;
    #else
      AT_ERROR("Cannot query CuDNN version if ATen_cuda is not built with CuDNN");
    #endif
        */
    }
    
    pub fn versioncudart(&self) -> i64 {
        
        todo!();
        /*
            #ifdef CUDART_VERSION
      return CUDART_VERSION;
    #else
      TORCH_CHECK(
        false,
        "Cannot query CUDART version because CUDART is not available");
    #endif
        */
    }
    
    pub fn hascudart(&self) -> bool {
        
        todo!();
        /*
            #ifdef CUDART_VERSION
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn show_config(&self) -> String {
        
        todo!();
        /*
            ostringstream oss;

      int runtimeVersion;
      cudaRuntimeGetVersion(&runtimeVersion);

      auto printCudaStyleVersion = [&](int v) {
        oss << (v / 1000) << "." << (v / 10 % 100);
        if (v % 10 != 0) {
          oss << "." << (v % 10);
        }
      };

    #ifndef __HIP_PLATFORM_HCC__
      oss << "  - CUDA Runtime ";
    #else
      oss << "  - HIP Runtime ";
    #endif
      printCudaStyleVersion(runtimeVersion);
      oss << "\n";

      // TODO: Make HIPIFY understand CUDART_VERSION macro
    #ifndef __HIP_PLATFORM_HCC__
      if (runtimeVersion != CUDART_VERSION) {
        oss << "  - Built with CUDA Runtime ";
        printCudaStyleVersion(CUDART_VERSION);
        oss << "\n";
      }
      oss << "  - NVCC architecture flags: " << NVCC_FLAGS_EXTRA << "\n";
    #endif

    #ifndef __HIP_PLATFORM_HCC__
    #if AT_CUDNN_ENABLED()

      auto printCudnnStyleVersion = [&](int v) {
        oss << (v / 1000) << "." << (v / 100 % 10);
        if (v % 100 != 0) {
          oss << "." << (v % 100);
        }
      };

      usize cudnnVersion = cudnnGetVersion();
      oss << "  - CuDNN ";
      printCudnnStyleVersion(cudnnVersion);
      usize cudnnCudartVersion = cudnnGetCudartVersion();
      if (cudnnCudartVersion != CUDART_VERSION) {
        oss << "  (built against CUDA ";
        printCudaStyleVersion(cudnnCudartVersion);
        oss << ")";
      }
      oss << "\n";
      if (cudnnVersion != CUDNN_VERSION) {
        oss << "    - Built with CuDNN ";
        printCudnnStyleVersion(CUDNN_VERSION);
        oss << "\n";
      }
    #endif
    #else
      // TODO: Check if miopen has the functions above and unify
      oss << "  - MIOpen " << MIOPEN_VERSION_MAJOR << "." << MIOPEN_VERSION_MINOR << "." << MIOPEN_VERSION_PATCH << "\n";
    #endif

    #ifdef USE_MAGMA
      oss << "  - Magma " << MAGMA_VERSION_MAJOR << "." << MAGMA_VERSION_MINOR << "." << MAGMA_VERSION_MICRO << "\n";
    #endif

      return oss.str();
        */
    }
    
    pub fn batchnorm_min_epsilon_cu_dnn(&self) -> f64 {
        
        todo!();
        /*
            #if AT_CUDNN_ENABLED()
      return CUDNN_BN_MIN_EPSILON;
    #else
      AT_ERROR(
          "Cannot query CUDNN_BN_MIN_EPSILON if ATen_cuda is not built with CuDNN");
    #endif
        */
    }
    
    pub fn cu_fft_get_plan_cache_max_size(&self, device_index: i64) -> i64 {
        
        todo!();
        /*
            return native::detail::cufft_get_plan_cache_max_size_impl(device_index);
        */
    }
    
    pub fn cu_fft_set_plan_cache_max_size(&self, 
        device_index: i64,
        max_size:     i64)  {
        
        todo!();
        /*
            native::detail::cufft_set_plan_cache_max_size_impl(device_index, max_size);
        */
    }
    
    pub fn cu_fft_get_plan_cache_size(&self, device_index: i64) -> i64 {
        
        todo!();
        /*
            return native::detail::cufft_get_plan_cache_size_impl(device_index);
        */
    }
    
    pub fn cu_fft_clear_plan_cache(&self, device_index: i64)  {
        
        todo!();
        /*
            native::detail::cufft_clear_plan_cache_impl(device_index);
        */
    }
    
    pub fn get_num_gpu_s(&self) -> i32 {
        
        todo!();
        /*
            return device_count();
        */
    }
    
    pub fn device_synchronize(&self, device_index: i64)  {
        
        todo!();
        /*
            DeviceGuard device_guard(Device(DeviceType_CUDA, device_index));
      device_synchronize();
        */
    }
}

/// Sigh, the registry doesn't support namespaces :(
register_cuda_hooks!(CUDAHooks);
