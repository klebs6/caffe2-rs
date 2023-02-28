/*!
  | NB: Class must live in `at` due to limitations
  | of Registry.h.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/detail/CUDAHooksInterface.h]

#[cfg(target_os = "windows")]
pub const CUDA_HELP: &'static str =
  "PyTorch splits its backend into two shared libraries: a CPU library \
  and a CUDA library; this error has occurred because you are trying \
  to use some CUDA functionality, but the CUDA library has not been \
  loaded by the dynamic linker for some reason.  The CUDA library MUST \
  be loaded, EVEN IF you don't directly use any symbols from the CUDA library! \
  One common culprit is a lack of -INCLUDE:?warp_size@cuda@at@@YAHXZ \
  in your link arguments; many dynamic linkers will delete dynamic library \
  dependencies if you don't depend on any of their symbols.  You can check \
  if this has occurred by using link on your binary to see if there is a \
  dependency on *_cuda.dll library.";

#[cfg(not(target_os = "windows"))]
pub const CUDA_HELP: &'static str =
  "PyTorch splits its backend into two shared libraries: a CPU library \
  and a CUDA library; this error has occurred because you are trying \
  to use some CUDA functionality, but the CUDA library has not been \
  loaded by the dynamic linker for some reason.  The CUDA library MUST \
  be loaded, EVEN IF you don't directly use any symbols from the CUDA library! \
  One common culprit is a lack of -Wl,--no-as-needed in your link arguments; many \
  dynamic linkers will delete dynamic library dependencies if you don't \
  depend on any of their symbols.  You can check if this has occurred by \
  using ldd on your binary to see if there is a dependency on *_cuda.so \
  library.";

/**
  | The CUDAHooksInterface is an omnibus interface
  | for any CUDA functionality which we may want to
  | call into from CPU code (and thus must be
  | dynamically dispatched, to allow for separate
  | compilation of CUDA code).  How do I decide if
  | a function should live in this class?  There
  | are two tests:
  |
  |  1. Does the *implementation* of this function
  |     require linking against CUDA libraries?
  |
  |  2. Is this function *called* from non-CUDA
  |  ATen code?
  |
  | (2) should filter out many ostensible
  | use-cases, since many times a CUDA function
  | provided by ATen is only really ever used by
  | actual CUDA code.
  |
  | TODO: Consider putting the stub definitions in
  | another class, so that one never forgets to
  | implement each virtual function in the real
  | implementation in CUDAHooks.  This probably
  | doesn't buy us much though.
  */
pub trait CUDAHooksInterface {

    /// Initialize THCState and, transitively, the
    /// CUDA state
    ///
    fn initcuda(&self) -> Box<THCState> {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot initialize CUDA without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn get_default_cuda_generator(&self, device_index: DeviceIndex) -> &Generator {
        let device_index: DeviceIndex = device_index.unwrap_or(-1);

        todo!();
        /*
            TORCH_CHECK(false, "Cannot get default CUDA generator without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn get_device_from_ptr(&self, data: *mut void) -> Device {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot get device of pointer on CUDA without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn is_pinned_ptr(&self, data: *mut void) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn hascuda(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn hascudart(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn hasmagma(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn has_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn nvrtc(&self) -> &NVRTC {
        
        todo!();
        /*
            TORCH_CHECK(false, "NVRTC requires CUDA. ", CUDA_HELP);
        */
    }
    
    fn current_device(&self) -> i64 {
        
        todo!();
        /*
            return -1;
        */
    }
    
    fn has_primary_context(&self, device_index: i64) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot call hasPrimaryContext(", device_index, ") without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn get_devce_index_with_primary_context(&self) -> Option<i64> {
        
        todo!();
        /*
            return nullopt;
        */
    }
    
    fn get_pinned_memory_allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
            TORCH_CHECK(false, "Pinned memory requires CUDA. ", CUDA_HELP);
        */
    }
    
    fn get_cuda_device_allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
            TORCH_CHECK(false, "CUDADeviceAllocator requires CUDA. ", CUDA_HELP);
        */
    }
    
    fn compiled_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn compiled_with_mio_pen(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn supports_dilated_convolution_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn supports_depthwise_convolution_with_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    fn version_cu_dnn(&self) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot query cuDNN version without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn versioncudart(&self) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot query CUDART version without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn show_config(&self) -> String {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot query detailed CUDA version without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn batchnorm_min_epsilon_cu_dnn(&self) -> f64 {
        
        todo!();
        /*
            TORCH_CHECK(false,
            "Cannot query batchnormMinEpsilonCuDNN() without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn cu_fft_get_plan_cache_max_size(&self, device_index: i64) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn cu_fft_set_plan_cache_max_size(&self, 
        device_index: i64,
        max_size:     i64)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn cu_fft_get_plan_cache_size(&self, device_index: i64) -> i64 {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn cu_fft_clear_plan_cache(&self, device_index: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
        */
    }
    
    fn get_num_gpu_s(&self) -> i32 {
        
        todo!();
        /*
            return 0;
        */
    }
    
    fn device_synchronize(&self, device_index: i64)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Cannot synchronize CUDA device without ATen_cuda library. ", CUDA_HELP);
        */
    }
}

/**
  | NB: dummy argument to suppress "ISO C++11
  | requires at least one argument for the "..." in
  | a variadic macro"
  |
  */
pub struct CUDAHooksArgs {}

c10_declare_registry!{
    CUDAHooksRegistry, 
    CUDAHooksInterface, 
    CUDAHooksArgs
}

#[macro_export] macro_rules! register_cuda_hooks {
    ($clsname:ident) => {
        /*
        
          C10_REGISTER_CLASS(CUDAHooksRegistry, clsname, clsname)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/detail/CUDAHooksInterface.cpp]

/**
  | NB: We purposely leak the CUDA hooks object.
  | This is because under some situations, we may
  | need to reference the CUDA hooks while running
  | destructors of objects which were constructed
  | *prior* to the first invocation of
  | getCUDAHooks.
  |
  | The example which precipitated this change was
  | the fused kernel cache in the JIT.  The kernel
  | cache is a global variable which caches both
  | CPU and CUDA kernels; CUDA kernels must
  | interact with CUDA hooks on destruction.
  |
  | Because the kernel cache handles CPU kernels
  | too, it can be constructed before we initialize
  | CUDA; if it contains CUDA kernels at program
  | destruction time, you will destruct the CUDA
  | kernels after CUDA hooks has been unloaded.
  |
  | In principle, we could have also fixed the
  | kernel cache store CUDA kernels in a separate
  | global variable, but this solution is much
  | simpler.
  |
  | CUDAHooks doesn't actually contain any data, so
  | leaking it is very benign; you're probably
  | losing only a word (the vptr in the allocated
  | object.)
  |
  */
lazy_static!{
    /*
    static CUDAHooksInterface* cuda_hooks = nullptr;
    */
}

pub fn get_cuda_hooks() -> &CUDAHooksInterface {
    
    todo!();
        /*
            // NB: The once_flag here implies that if you try to call any CUDA
      // functionality before libATen_cuda.so is loaded, CUDA is permanently
      // disabled for that copy of ATen.  In principle, we can relax this
      // restriction, but you might have to fix some code.  See getVariableHooks()
      // for an example where we relax this restriction (but if you try to avoid
      // needing a lock, be careful; it doesn't look like Registry.h is thread
      // safe...)
    #if !defined C10_MOBILE
      static once_flag once;
      call_once(once, [] {
        cuda_hooks = CUDAHooksRegistry()->Create("CUDAHooks", CUDAHooksArgs{}).release();
        if (!cuda_hooks) {
          cuda_hooks = new CUDAHooksInterface();
        }
      });
    #else
      if (cuda_hooks == nullptr) {
        cuda_hooks = new CUDAHooksInterface();
      }
    #endif
      return *cuda_hooks;
        */
}

c10_define_registry!{
    CUDAHooksRegistry, 
    CUDAHooksInterface, 
    CUDAHooksArgs
}
