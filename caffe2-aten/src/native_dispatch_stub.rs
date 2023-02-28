/*!
  | Implements instruction set specific function
  | dispatch.
  |
  | Kernels that may make use of specialized
  | instruction sets (e.g. AVX) are compiled
  | multiple times with different compiler flags
  | (e.g. -mavx). A DispatchStub contains a table
  | of function pointers for a kernel. At runtime,
  | the fastest available kernel is chosen based on
  | the features reported by cpuinfo.
  |
  | Example:
  |
  | In native/MyKernel.h:
  |   using fn_type = void(*)(const Tensor& x);
  |   DECLARE_DISPATCH(fn_type, stub);
  |
  | In native/MyKernel.cpp
  |   DEFINE_DISPATCH(stub);
  |
  | In native/cpu/MyKernel.cpp:
  |   namespace {
  |     // use anonymous namespace so that different cpu versions won't conflict
  |     void kernel(const Tensor& x) { ... }
  |   }
  |   register_dispatch(stub, &kernel);
  |
  | To call:
  |   stub(kCPU, tensor);
  |
  | TODO: CPU instruction set selection should be folded into whatever
  | the main dispatch mechanism is.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/DispatchStub.h]

/**
  | ignore warnings about DispatchStub::DEFAULT,
  | 
  | AVX, AVX2 defined elsewhere
  |
  */
pub enum CPUCapability {

    DEFAULT = 0,

    #[cfg(HAVE_VSX_CPU_DEFINITION)]      VSX  = 1,

    #[cfg(not(HAVE_VSX_CPU_DEFINITION))] AVX  = 1,
    #[cfg(not(HAVE_VSX_CPU_DEFINITION))] AVX2 = 2,

    NUM_OPTIONS
}


pub struct DispatchStub<FnPtr, T> {}

/**
  | The sole purpose of this class is to outline
  | methods that don't need to be specialized
  | or otherwise inlined and duplicated
  | (by the compiler due to template expansion),
  | since it causes size bloat if there are
  | a significant number of specialization
  | of the DispatchStub<> class.
  |
  */
pub struct DispatchStubImpl {

    /**
      | Fixing dispatch error in Windows debug
      | builds.
      |
      | See
      | https://github.com/pytorch/pytorch/issues/22681
      | for more details.
      */
    #[cfg(all(target_os = "windows",debug_assertions))]      cpu_dispatch_ptr:  Atomic<*mut void>,
    #[cfg(all(target_os = "windows",debug_assertions))]      cuda_dispatch_ptr: *mut void,
    #[cfg(all(target_os = "windows",debug_assertions))]      hip_dispatch_ptr:  *mut void,

    #[cfg(not(all(target_os = "windows",debug_assertions)))] cpu_dispatch_ptr:  Atomic<*mut void>, // default = { nullptr }
    #[cfg(not(all(target_os = "windows",debug_assertions)))] cuda_dispatch_ptr: *mut void, // default = nullptr
    #[cfg(not(all(target_os = "windows",debug_assertions)))] hip_dispatch_ptr:  *mut void, // default = nullptr
}

impl DispatchStubImpl {
    
    pub fn get_call_ptr(&mut self, 
        device_type: DeviceType,
        default:     *mut void,

        #[cfg(HAVE_AVX_CPU_DEFINITION)]
        avx:         *mut void,

        #[cfg(HAVE_AVX2_CPU_DEFINITION)]
        avx2:        *mut void,

        #[cfg(HAVE_VSX_CPU_DEFINITION)]
        vsx:         *mut void)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | The CPU Dispatch actual method is chosen
      | in decreasing order of preference by
      | 
      | DispatchStubImpl::choose_cpu_impl()
      | in case none is found by
      | 
      | DispatchStubImpl::get_call_ptr()
      | in cpu_dispatch_ptr.
      |
      */
    pub fn choose_cpu_impl(&mut self, 
        default: *mut void,

        #[cfg(HAVE_AVX_CPU_DEFINITION)]
        avx:     *mut void,

        #[cfg(HAVE_AVX2_CPU_DEFINITION)]
        avx2:    *mut void,

        #[cfg(HAVE_VSX_CPU_DEFINITION)]
        vsx:     *mut void)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_call_ptr(&mut self, 
        device_type: DeviceType,
        default:     *mut void,

        #[cfg(HAVE_AVX_CPU_DEFINITION)]
        avx:         *mut void,

        #[cfg(HAVE_AVX2_CPU_DEFINITION)]
        avx2:        *mut void,

        #[cfg(HAVE_VSX_CPU_DEFINITION)]
        vsx:         *mut void)  {
        
        todo!();
        /*
            switch (device_type) {
        case DeviceType::CPU: {
          // Use memory_order_relaxed here since even if two threads race,
          // they will still compute the same value for cpu_dispatch_ptr.
          auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
          if (!fptr) {
            fptr = choose_cpu_impl(
              DEFAULT
    #ifdef HAVE_AVX_CPU_DEFINITION
              , AVX
    #endif
    #ifdef HAVE_AVX2_CPU_DEFINITION
              , AVX2
    #endif
    #ifdef HAVE_VSX_CPU_DEFINITION
              , VSX
    #endif
            );
            cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
          }
          return fptr;
        }

        case DeviceType::CUDA:
          TORCH_INTERNAL_ASSERT(cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
          return cuda_dispatch_ptr;

        case DeviceType::HIP:
          TORCH_INTERNAL_ASSERT(hip_dispatch_ptr, "DispatchStub: missing HIP kernel");
          return hip_dispatch_ptr;

        default:
          AT_ERROR("DispatchStub: unsupported device type", device_type);
      }
        */
    }
    
    pub fn choose_cpu_impl(&mut self, 
        default: *mut void,

        #[cfg(HAVE_AVX_CPU_DEFINITION)]
        avx:     *mut void,

        #[cfg(HAVE_AVX2_CPU_DEFINITION)]
        avx2:    *mut void,

        #[cfg(HAVE_VSX_CPU_DEFINITION)]
        vsx:     *mut void)  {
        
        todo!();
        /*
            auto capability = static_cast<int>(get_cpu_capability());
      (void)capability;
    #ifdef HAVE_AVX2_CPU_DEFINITION
      if (capability >= static_cast<int>(CPUCapability::AVX2)) {
        TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
        return AVX2;
      }
    #endif
    #ifdef HAVE_AVX_CPU_DEFINITION
      if (capability >= static_cast<int>(CPUCapability::AVX)) {
        TORCH_INTERNAL_ASSERT(AVX, "DispatchStub: missing AVX kernel");
        return AVX;
      }
    #endif
    #ifdef HAVE_VSX_CPU_DEFINITION
      if (capability >= static_cast<int>(CPUCapability::VSX)) {
        TORCH_INTERNAL_ASSERT(VSX, "DispatchStub: missing VSX kernel");
        return VSX;
      }
    #endif
      TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
      return DEFAULT;
        */
    }
}

lazy_static!{
    /*
    template <typename rT, typename T, typename... Args>
    struct DispatchStub<rT (*)(Args...), T> {
      using FnPtr = rT (*) (Args...);

      DispatchStub() = default;
      DispatchStub(const DispatchStub&) = delete;
      DispatchStub& operator=(const DispatchStub&) = delete;

    private:
      FnPtr get_call_ptr(DeviceType device_type) {
        return reinterpret_cast<FnPtr>(
          impl.get_call_ptr(device_type
          , reinterpret_cast<void*>(DEFAULT)
    #ifdef HAVE_AVX_CPU_DEFINITION
          , reinterpret_cast<void*>(AVX)
    #endif
    #ifdef HAVE_AVX2_CPU_DEFINITION
          , reinterpret_cast<void*>(AVX2)
    #endif
    #ifdef HAVE_VSX_CPU_DEFINITION
          , reinterpret_cast<void*>(VSX)
    #endif
          )
        );
      }

    public:
      template <typename... ArgTypes>
      rT operator()(DeviceType device_type, ArgTypes&&... args) {
        FnPtr call_ptr = get_call_ptr(device_type);
        return (*call_ptr)(std::forward<ArgTypes>(args)...);
      }

      void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
        impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
      }

      void set_hip_dispatch_ptr(FnPtr fn_ptr) {
        impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
      }

      static FnPtr DEFAULT;
    #ifdef HAVE_AVX_CPU_DEFINITION
      static FnPtr AVX;
    #endif
    #ifdef HAVE_AVX2_CPU_DEFINITION
      static FnPtr AVX2;
    #endif
    #ifdef HAVE_VSX_CPU_DEFINITION
      static FnPtr VSX;
    #endif
    private:
      DispatchStubImpl impl;
    };
    */
}

pub struct RegisterCUDADispatch<FnPtr,T> {

}

impl RegisterCUDADispatch<FnPtr,T> {
    
    pub fn new(
        stub:  &mut DispatchStub<FnPtr,T>,
        value: FnPtr) -> Self {
    
        todo!();
        /*


            stub.set_cuda_dispatch_ptr(value);
        */
    }
}

//----------------------------
pub struct RegisterHIPDispatch<FnPtr,T> {

}

impl RegisterHIPDispatch<FnPtr,T> {
    
    pub fn new(
        stub:  &mut DispatchStub<FnPtr,T>,
        value: FnPtr) -> Self {
    
        todo!();
        /*


            // TODO: make this point at hip_dispatch_ptr
        stub.set_cuda_dispatch_ptr(value);
        */
    }
}

/**
  | Compiler will complain if you put things like
  | std::tuple<Tensor, Tensor> in the `fn` argument
  | of DECLARE_DISPATCH. Some possible workarounds,
  | e.g., adding parentheses and using helper
  | struct to get rid of the parentheses, do not
  | work with MSVC.
  |
  | So do a `using`-declaration if you need to pass
  | in such `fn`, e.g.,
  | grid_sampler_2d_backward_cpu_kernel in
  | GridSampleKernel.h.
  */
#[macro_export] macro_rules! declare_dispatch {
    ($fn:ty, $name:expr) => {
        /*
        
          struct name : DispatchStub<fn, name> {   
            name() = default;                      
            name(const name&) = delete;            
            name& operator=(const name&) = delete; 
          };                                       
          extern TORCH_API struct name name
        */
    }
}

#[macro_export] macro_rules! define_dispatch {
    ($name:expr) => {
        /*
                struct name name
        */
    }
}

#[macro_export] macro_rules! register_arch_dispatch {
    ($name:expr, $arch:expr, $fn:expr) => {
        /*
        
          template <> decltype(fn) DispatchStub<decltype(fn), struct name>::arch = fn;
        */
    }
}

#[cfg(HAVE_AVX_CPU_DEFINITION)]
#[macro_export] macro_rules! register_avx_dispatch {
    ($name:expr, $fn:expr) => {
        /*
                REGISTER_ARCH_DISPATCH(name, AVX, fn)
        */
    }
}

#[cfg(not(HAVE_AVX_CPU_DEFINITION))]
#[macro_export] macro_rules! register_avx_dispatch { ($name:expr, $fn:expr) => { } }

#[cfg(HAVE_AVX2_CPU_DEFINITION)]
#[macro_export] macro_rules! register_avx2_dispatch {
    ($name:expr, $fn:expr) => {
        /*
                REGISTER_ARCH_DISPATCH(name, AVX2, fn)
        */
    }
}

#[cfg(not(HAVE_AVX2_CPU_DEFINITION))]
#[macro_export] macro_rules! register_avx2_dispatch { ($name:expr, $fn:expr) => { } }

#[cfg(HAVE_VSX_CPU_DEFINITION)]
#[macro_export] macro_rules! register_vsx_dispatch {
    ($name:expr, $fn:expr) => {
        /*
                REGISTER_ARCH_DISPATCH(name, VSX, fn)
        */
    }
}

#[cfg(not(HAVE_VSX_CPU_DEFINITION))]
#[macro_export] macro_rules! register_vsx_dispatch { ($name:expr, $fn:expr) => { } }

#[macro_export] macro_rules! register_no_cpu_dispatch {
    ($name:expr, $fn_type:expr) => {
        /*
        
          REGISTER_ARCH_DISPATCH(name, DEFAULT, static_cast<fn_type>(nullptr))         
          REGISTER_AVX_DISPATCH(name, static_cast<fn_type>(nullptr))                   
          REGISTER_AVX2_DISPATCH(name, static_cast<fn_type>(nullptr))          
          REGISTER_VSX_DISPATCH(name, static_cast<fn_type>(nullptr))
        */
    }
}

#[macro_export] macro_rules! register_cuda_dispatch {
    ($name:expr, $fn:expr) => {
        /*
        
          static RegisterCUDADispatch<decltype(fn), struct name> name ## __register(name, fn);
        */
    }
}

#[macro_export] macro_rules! register_hip_dispatch {
    ($name:expr, $fn:expr) => {
        /*
        
          static RegisterHIPDispatch<decltype(fn), struct name> name ## __register(name, fn);
        */
    }
}

/**
  | NB: This macro must be used in an actual 'cu'
  | file; if you try using it from a 'cpp' file it
  | will not work!
  |
  */
#[cfg(__CUDACC__)]
#[macro_export] macro_rules! register_dispatch {
    ($name:expr, $fn:expr) => {
        /*
                REGISTER_CUDA_DISPATCH(name, fn)
        */
    }
}

/**
  | TODO: cut this over to HIP dispatch once
  | we stop pretending that CUDA is HIP in
  | the PyTorch
  | 
  | HIPify build.
  |
  */
#[cfg(__HIPCC__)]
#[macro_export] macro_rules! register_dispatch {
    ($name:expr, $fn:expr) => {
        /*
                REGISTER_CUDA_DISPATCH(name, fn)
        */
    }
}

// #define register_dispatch(name, fn) REGISTER_HIP_DISPATCH(name, fn)
#[cfg(CPU_CAPABILITY)]
#[macro_export] macro_rules! register_dispatch {
    ($name:expr, $fn:expr) => {
        /*
                REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
        */
    }
}

#[macro_export] macro_rules! register_dispatch {
    ($name:expr, $fn:expr) => { 
        //TODO: one of the above cfg vars needs to
        //be set
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/DispatchStub.cpp]

pub fn compute_cpu_capability() -> CPUCapability {
    
    todo!();
        /*
            auto envar = std::getenv("ATEN_CPU_CAPABILITY");
      if (envar) {
    #ifdef HAVE_VSX_CPU_DEFINITION
        if (strcmp(envar, "vsx") == 0) {
          return CPUCapability::VSX;
        }
    #else
        if (strcmp(envar, "avx2") == 0) {
          return CPUCapability::AVX2;
        }
        if (strcmp(envar, "avx") == 0) {
          return CPUCapability::AVX;
        }
    #endif
        if (strcmp(envar, "default") == 0) {
          return CPUCapability::DEFAULT;
        }
        TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
      }

    #if !defined(__powerpc__) && !defined(__s390x__)
      if (cpuinfo_initialize()) {
        if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
          return CPUCapability::AVX2;
        }
        if (cpuinfo_has_x86_avx()) {
          return CPUCapability::AVX;
        }
      }
    #endif
    #ifdef HAVE_VSX_CPU_DEFINITION
      return CPUCapability::VSX;
    #else
      return CPUCapability::DEFAULT;
    #endif
        */
}

pub fn get_cpu_capability() -> CPUCapability {
    
    todo!();
        /*
            static CPUCapability capability = compute_cpu_capability();
      return capability;
        */
}
