crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Context.h]

pub struct Context {
    thc_init:                       OnceFlag,
    thh_init:                       OnceFlag,
    enabled_cudnn:                  bool, // default = true
    deterministic_cudnn:            bool, // default = false
    deterministic_algorithms:       bool, // default = false
    benchmark_cudnn:                bool, // default = false
    allow_tf32_cudnn:               bool, // default = true
    allow_tf32_cublas:              bool, // default = true
    enabled_mkldnn:                 bool, // default = true

    #[cfg(C10_MOBILE)]
    release_original_weights:       bool, // default = true

    #[cfg(not(C10_MOBILE))]
    release_original_weights:       bool, // default = false

    display_vmap_fallback_warnings: bool, // default = false
    quantized_engine:               Option<QEngine>, // default = nullopt

    thc_state:                      Box<THCState, fn(*mut THCState) -> ()>,
    thh_state:                      Box<THHState, fn(*mut THHState) -> ()>,

    prev_allocator_ptr:             *mut Allocator, // default = { nullptr }
}

impl Context {
    
    pub fn default_generator(&mut self, device: Device) -> &Generator {
        
        todo!();
        /*
            DeviceType device_type = device.type();
        initCUDAIfNeeded(device_type);
        initHIPIfNeeded(device_type);
        if (device_type == at::kCPU) {
          return at::detail::getDefaultCPUGenerator();
        } else if (device_type == at::kCUDA) {
          return at::detail::getCUDAHooks().getDefaultCUDAGenerator(device.index());
        } else {
          AT_ERROR(DeviceTypeName(device_type), " device type not enabled.");
        }
        */
    }
    
    pub fn get_device_from_ptr(&mut self, 
        data:        *mut void,
        device_type: DeviceType) -> Device {
        
        todo!();
        /*
            initCUDAIfNeeded(device_type);
        initHIPIfNeeded(device_type);
        if (device_type == at::kCPU) {
          return DeviceType::CPU;
        } else if (device_type == at::kCUDA) {
          return at::detail::getCUDAHooks().getDeviceFromPtr(data);
        } else {
          AT_ERROR(DeviceTypeName(device_type), " device type not enabled.");
        }
        */
    }
    
    pub fn is_pinned_ptr(data: *mut void) -> bool {
        
        todo!();
        /*
            return detail::getCUDAHooks().isPinnedPtr(data);
        */
    }
    
    pub fn has_openmp() -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn has_mkl() -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn haslapack() -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn hasmkldnn() -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn hasmagma() -> bool {
        
        todo!();
        /*
            return detail::getCUDAHooks().hasMAGMA();
        */
    }
    
    pub fn hascuda() -> bool {
        
        todo!();
        /*
            return detail::getCUDAHooks().hasCUDA();
        */
    }
    
    pub fn hascudart() -> bool {
        
        todo!();
        /*
            return detail::getCUDAHooks().hasCUDART();
        */
    }
    
    pub fn versioncudart() -> i64 {
        
        todo!();
        /*
            return detail::getCUDAHooks().versionCUDART();
        */
    }
    
    pub fn has_hip() -> bool {
        
        todo!();
        /*
            return detail::getHIPHooks().hasHIP();
        */
    }
    
    pub fn has_xla() -> bool {
        
        todo!();
        /*
            return c10::impl::hasDeviceGuardImpl(at::DeviceType::XLA);
        */
    }
    
    pub fn has_mlc() -> bool {
        
        todo!();
        /*
            return c10::impl::hasDeviceGuardImpl(at::DeviceType::MLC);
        */
    }

    /**
      | defined in header so that getNonVariableType
      | has ability to inline call_once
      |   check. getNonVariableType is called fairly
      |   frequently
      */
    pub fn lazy_initcuda(&mut self) -> *mut THCState {
        
        todo!();
        /*
            std::call_once(thc_init,[&] {
          thc_state = detail::getCUDAHooks().initCUDA();
        });
        return thc_state.get();
        */
    }
    
    pub fn lazy_init_hip(&mut self) -> *mut THHState {
        
        todo!();
        /*
            std::call_once(thh_init,[&] {
          thh_state = detail::getHIPHooks().initHIP();
        });
        return thh_state.get();
        */
    }
    
    pub fn getnvrtc<'a>() -> &'a NVRTC {
        
        todo!();
        /*
            return detail::getCUDAHooks().nvrtc();
        */
    }
    
    pub fn get_thc_state(&mut self) -> *mut THCState {
        
        todo!();
        /*
            // AT_ASSERT(thc_state);
        return thc_state.get();
        */
    }
    
    pub fn get_thh_state(&mut self) -> *mut THHState {
        
        todo!();
        /*
            return thh_state.get();
        */
    }
    
    pub fn set_flush_denormal(on: bool) -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | NB: This method is *purely* whether or not
      | a user requested that CuDNN was enabled, it
      |   doesn't actually say anything about whether
      |   or not CuDNN is actually usable.  Use
      |   cudnn_is_acceptable to test this instead
      */
    pub fn user_enabled_cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_user_enabled_cu_dnn(&mut self, e: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn user_enabled_mkldnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_user_enabled_mkldnn(&mut self, e: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn benchmark_cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_benchmark_cu_dnn(&mut self, _0: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn deterministic_cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_deterministic_cu_dnn(&mut self, _0: bool)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Note [Enabling Deterministic Operations]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |
      | Operations in PyTorch that normally act
      | nondeterministically, but have an alternate
      |   deterministic implementation, should satisfy
      |   the following requirements:
      |
      | * Include this comment: "See Note [Enabling
      | Deterministic Operations]"
      |
      | * Check the value of
      |   `at::globalContext().deterministicAlgorithms()`
      |     to toggle between nondeterministic and
      |     deterministic implementations.
      |
      | * Have an entry in the list of PyTorch
      |   operations that toggle between
      |     nondeterministic and deterministic
      |     implementations, in the docstring of
      |     `use_deterministic_algorithms()` in
      |     torch/__init__.py
      |
      | `example_func()` below shows an example of
      | toggling between nondeterministic and
      |   deterministic implementations:
      |
      |    void example_func() {
      |      // See Note [Enabling Deterministic Operations]
      |      if (at::globalContext().deterministicAlgorithms()) {
      |        example_func_deterministic();
      |      } else {
      |        example_func_nondeterministic();
      |      }
      |    }
      */
    pub fn deterministic_algorithms(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_deterministic_algorithms(&mut self, _0: bool)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Note [Writing Nondeterministic Operations]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      | Operations in PyTorch that act
      | nondeterministically and do not have an
      | alternate deterministic implementation should
      | satisfy the following requirements:
      |
      | * Include this comment: "See Note [Writing
      | Nondeterministic Operations]"
      |
      | * Include a comment explaining why the
      | operation is nondeterministic.
      |
      | * Throw an error when
      |   `Context::deterministicAlgorithms()` is
      |     true. Most of the time, this should be
      |     accomplished by calling
      |     `at::globalContext().alertNotDeterminstic()`.
      |     However, if the nondeterministic behavior
      |     is caused by the CuBLAS workspace
      |     configuration in CUDA >= 10.2,
      |     `at::globalContext().alertCuBLASConfigNotDeterministic()`
      |     should be called instead (in this case,
      |     a comment explaining why the operation is
      |     nondeterministic is not necessary). See
      |     below for details on these methods.
      |
      | * Have an entry in the list of
      |   nondeterministic PyTorch operations in the
      |     docstring of
      |     `use_deterministic_algorithms()` in
      |     torch/__init__.py
      |
      | * Have a test function in
      |   `test/test_torch.py` whose name begins with
      |     `test_nondeterministic_alert_`. Alternatively,
      |     if CuBLAS workspace configuration is the
      |     reason for nondeterminism, the operation
      |     should be included in the
      |     `test_cublas_config_nondeterministic_alert`
      |     test. Any new tests should ideally follow
      |     a pattern similar to the existing ones.
      |
      | `example_func()` below shows an example of
      | the comments and error-throwing code for
      |   a nondeterministic operation:
      |
      |    void example_func() {
      |      // See Note [Writing Nondeterministic Operations]
      |      // Nondeterministic because <reason>
      |      at::globalContext().alertNondeterministic("example_func");
      |      ...
      |    }
      */

    /**
      | Throws an error if `Context::deterministicAlgorithms()`
      | is true
      |
      */
    pub fn alert_not_deterministic(caller: &str)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Throws an error if
      | `Context::deterministicAlgorithms()` is true,
      |   CUDA >= 10.2, and CUBLAS_WORKSPACE_CONFIG is
      |   not set to either ":16:8" or ":4096:8". For
      |   more details:
      |   https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
      */
    pub fn alert_cu_blas_config_not_deterministic(&self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn allow_tf32cu_dnn(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_allow_tf32cu_dnn(&mut self, _0: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn allow_tf32cublas(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_allow_tf32cublas(&mut self, _0: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn q_engine(&self) -> QEngine {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_qengine(&mut self, e: QEngine)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn supported_qengines<'a>() -> &'a Vec<QEngine> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_xnnpack_available() -> bool {
        
        todo!();
        /*
        
        */
    }

    /**
      | This method is used to release the original
      | weight after pre-packing.
      |
      | It should be called once before
      | loading/running the model.
      |
      | NB: By default it is set to true for mobile
      | builds.
      */
    pub fn set_release_weights_when_prepacking(&mut self, e: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn release_weights_when_prepacking(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_display_vmap_fallback_warnings(&mut self, enabled: bool)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn are_vmap_fallback_warnings_enabled(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_default_mobile_cpu_allocator(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn unset_default_mobile_cpu_allocator(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn init_cuda_if_needed(&mut self, p: DeviceType)  {
        
        todo!();
        /*
            if (p == DeviceType::CUDA) {
          lazyInitCUDA();
        }
        */
    }
    
    pub fn init_hip_if_needed(&mut self, p: DeviceType)  {
        
        todo!();
        /*
            if (p == DeviceType::HIP) {
          lazyInitHIP();
        }
        */
    }
    
    pub fn check_cu_blas_config_deterministic() -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new() -> Self {
    
        todo!();
        /*
            : thc_state(nullptr, [](THCState* p) { /* no-op */ }),
          thh_state(nullptr, [](THHState* p) { /* no-op */ })
        */
    }
    
    /**
     | NB: This method is *purely* whether or not
     | a user requested that CuDNN was enabled, it
     | doesn't actually say anything about whether or
     | not CuDNN is actually usable.
     */
    pub fn user_enabled_cu_dnn(&self) -> bool {

        todo!();
        /*
            return enabled_cudnn;
        */
    }
    
    pub fn set_user_enabled_cu_dnn(&mut self, e: bool)  {
        
        todo!();
        /*
            enabled_cudnn = e;
        */
    }
    
    pub fn user_enabled_mkldnn(&self) -> bool {
        
        todo!();
        /*
            return enabled_mkldnn;
        */
    }
    
    pub fn set_user_enabled_mkldnn(&mut self, e: bool)  {
        
        todo!();
        /*
            enabled_mkldnn = e;
        */
    }
    
    pub fn deterministic_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return deterministic_cudnn;
        */
    }
    
    pub fn set_deterministic_cu_dnn(&mut self, b: bool)  {
        
        todo!();
        /*
            deterministic_cudnn = b;
        */
    }
    
    pub fn deterministic_algorithms(&self) -> bool {
        
        todo!();
        /*
            return _deterministic_algorithms;
        */
    }
    
    pub fn set_deterministic_algorithms(&mut self, b: bool)  {
        
        todo!();
        /*
            _deterministic_algorithms = b;
        */
    }
    
    pub fn alert_not_deterministic(&mut self, caller: &str)  {
        
        todo!();
        /*
            if (globalContext().deterministicAlgorithms()) {
        TORCH_CHECK(false,
          caller, " does not have a deterministic implementation, but you set "
          "'torch.use_deterministic_algorithms(True)'. You can turn off determinism ",
          "just for this operation if that's acceptable for your application. You "
          "can also file an issue at https://github.com/pytorch/pytorch/issues "
          "to help us prioritize adding deterministic support for this operation.");
      }
        */
    }
    
    pub fn allow_tf32cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return allow_tf32_cudnn;
        */
    }
    
    pub fn set_allow_tf32cu_dnn(&mut self, b: bool)  {
        
        todo!();
        /*
            allow_tf32_cudnn = b;
        */
    }
    
    pub fn check_cu_blas_config_deterministic(&mut self) -> bool {
        
        todo!();
        /*
            bool cublas_config_deterministic = true;
      // If using CUDA 10.2 or greater, need to make sure CuBLAS workspace config
      // is set to deterministic setting
      if (hasCUDART() && (versionCUDART() >= 10020)) {
        char* workspace_config = std::getenv(cublas_config_var_name);
        cublas_config_deterministic = (workspace_config != nullptr) && (
          (strcmp(workspace_config, cublas_deterministic_configs[0]) == 0)
          || (strcmp(workspace_config, cublas_deterministic_configs[1]) == 0)
        );
      }
      return cublas_config_deterministic;
        */
    }
    
    pub fn alert_cu_blas_config_not_deterministic(&self)  {
        
        todo!();
        /*
            static bool cublas_config_deterministic = checkCuBLASConfigDeterministic();
      TORCH_CHECK(!deterministicAlgorithms() || cublas_config_deterministic,
        "Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or ",
        "`at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because ",
        "it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this ",
        "case, you must set an environment variable before running your PyTorch application: ",
        cublas_config_var_name, "=", cublas_deterministic_configs[0], " or ",
        cublas_config_var_name, "=", cublas_deterministic_configs[1], ". For more information, go to ",
        "https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility"
      );
        */
    }
    
    pub fn benchmark_cu_dnn(&self) -> bool {
        
        todo!();
        /*
            return benchmark_cudnn;
        */
    }
    
    pub fn set_benchmark_cu_dnn(&mut self, b: bool)  {
        
        todo!();
        /*
            benchmark_cudnn = b;
        */
    }
    
    pub fn allow_tf32cublas(&self) -> bool {
        
        todo!();
        /*
            return allow_tf32_cublas;
        */
    }
    
    pub fn set_allow_tf32cublas(&mut self, b: bool)  {
        
        todo!();
        /*
            allow_tf32_cublas = b;
        */
    }
    
    pub fn has_mkl(&mut self) -> bool {
        
        todo!();
        /*
            #if AT_MKL_ENABLED()
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn hasmkldnn(&mut self) -> bool {
        
        todo!();
        /*
            #if AT_MKLDNN_ENABLED()
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn has_openmp(&mut self) -> bool {
        
        todo!();
        /*
            #ifdef _OPENMP
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn haslapack(&mut self) -> bool {
        
        todo!();
        /*
            #ifdef USE_LAPACK
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn q_engine(&self) -> QEngine {
        
        todo!();
        /*
            // If wasn't explicitly set - take the last one available
      return quantized_engine.value_or(supportedQEngines().back());
        */
    }
    
    pub fn set_qengine(&mut self, e: QEngine)  {
        
        todo!();
        /*
            const auto& qengines = supportedQEngines();
      if (std::find(qengines.begin(), qengines.end(), e) != qengines.end()) {
        quantized_engine = e;
        return;
      }
      TORCH_CHECK(false, "quantized engine ", toString(e), " is not supported");
        */
    }
    
    pub fn supported_qengines(&mut self) -> &Vec<QEngine> {
        
        todo!();
        /*
            static auto supported_qengines = []() {
        std::vector<at::QEngine> engines = {};
        // Engines are listed in priority order: later one wins
        // By default we prefer FBGEMM if we're running on server side
        // QNNPACK on server side has some issue, so we disable it by default.
    #ifdef C10_MOBILE
        engines.push_back(at::kNoQEngine);
    #ifdef USE_PYTORCH_QNNPACK
        engines.push_back(at::kQNNPACK);
    #endif
    #else  // C10_MOBILE
    #ifdef USE_PYTORCH_QNNPACK
        engines.push_back(at::kQNNPACK);
    #endif
        engines.push_back(at::kNoQEngine);
    #endif // C10_MOBILE

    #ifdef USE_FBGEMM
        if (fbgemm::fbgemmSupportedCPU()) {
          engines.push_back(at::kFBGEMM);
        }
    #endif
        return engines;
      }();
      return supported_qengines;
        */
    }
    
    pub fn is_xnnpack_available(&mut self) -> bool {
        
        todo!();
        /*
            #ifdef USE_XNNPACK
      return true;
    #else
      return false;
    #endif
        */
    }
    
    pub fn release_weights_when_prepacking(&self) -> bool {
        
        todo!();
        /*
            return release_original_weights;
        */
    }
    
    pub fn set_release_weights_when_prepacking(&mut self, e: bool)  {
        
        todo!();
        /*
            release_original_weights = e;
        */
    }
    
    pub fn set_flush_denormal(&mut self, on: bool) -> bool {
        
        todo!();
        /*
            return at::cpu::set_flush_denormal(on);
        */
    }
    
    pub fn are_vmap_fallback_warnings_enabled(&self) -> bool {
        
        todo!();
        /*
            return display_vmap_fallback_warnings_;
        */
    }
    
    pub fn set_display_vmap_fallback_warnings(&mut self, enabled: bool)  {
        
        todo!();
        /*
            display_vmap_fallback_warnings_ = enabled;
        */
    }
    
    pub fn set_default_mobile_cpu_allocator(&mut self)  {
        
        todo!();
        /*
            TORCH_CHECK(prev_allocator_ptr_ == nullptr,
          "Already within the scope of another non-default cpu allocator."
          "Cannot set another allocator.");
      // Setting the priority high to make sure no other allocator gets used instead of this.
      prev_allocator_ptr_ = c10::GetCPUAllocator();
      c10::SetCPUAllocator(c10::GetDefaultMobileCPUAllocator(), /*priority*/ 100);
        */
    }
    
    pub fn unset_default_mobile_cpu_allocator(&mut self)  {
        
        todo!();
        /*
            TORCH_CHECK(prev_allocator_ptr_ != nullptr,
          "setDefaultMobileCPUAllocator must have been called "
          "before unsetDefaultMobileCPUAllocator.");
      // Setting the priority high to make sure no other allocator gets used instead of this.
      c10::SetCPUAllocator(prev_allocator_ptr_ , /*priority*/ 100);
      prev_allocator_ptr_ = nullptr;
        */
    }
}

#[inline] pub fn init()  {
    
    todo!();
        /*
            globalContext();
        */
}

#[inline] pub fn get_deprecated_type_properties<'a>(
        p: Backend,
        s: ScalarType) -> &'a mut DeprecatedTypeProperties {
    
    todo!();
        /*
            return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
          p, s);
        */
}

#[inline] pub fn cpu<'a>(s: ScalarType) -> &'a mut DeprecatedTypeProperties {
    
    todo!();
        /*
            return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
          Backend::CPU, s);
        */
}

#[inline] pub fn cuda<'a>(s: ScalarType) -> &'a mut DeprecatedTypeProperties {
    
    todo!();
        /*
            return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
          Backend::CUDA, s);
        */
}

#[inline] pub fn hip<'a>(s: ScalarType) -> &'a mut DeprecatedTypeProperties {
    
    todo!();
        /*
            return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
          Backend::HIP, s);
        */
}

#[inline] pub fn hascuda() -> bool {
    
    todo!();
        /*
            return globalContext().hasCUDA();
        */
}

#[inline] pub fn has_hip() -> bool {
    
    todo!();
        /*
            return globalContext().hasHIP();
        */
}

#[inline] pub fn has_xla() -> bool {
    
    todo!();
        /*
            return globalContext().hasXLA();
        */
}

#[inline] pub fn has_mlc() -> bool {
    
    todo!();
        /*
            return globalContext().hasMLC();
        */
}

/**
  | Despite its name, this function returns
  | the number of *CUDA* GPUs.
  |
  */
#[inline] pub fn get_num_gpu_s() -> usize {
    
    todo!();
        /*
            // WARNING: DO NOT ADD LOGIC TO HANDLE OTHER DEVICE TYPES TO THIS
      // FUNCTION.  If you are interested in interrogating the number of
      // devices for a specific device type, add that function to the
      // relevant library (e.g., similar to at::cuda::device_count())
      if (hasCUDA() && hasHIP()) {
        throw std::runtime_error(
            "Enabling both CUDA and HIP in ATen is not supported, as HIP masquerades "
            "to be CUDA (e.g., when you say CUDA, on a HIP build of ATen, this actually "
            "means HIP.  Rebuild PyTorch with one or the other disabled.");
      } else if (hasCUDA()) {
        return detail::getCUDAHooks().getNumGPUs();
      } else if (hasHIP()) {
        return detail::getHIPHooks().getNumGPUs();
      } else {
        return 0;
      }
        */
}

#[inline] pub fn has_openmp() -> bool {
    
    todo!();
        /*
            return globalContext().hasOpenMP();
        */
}

#[inline] pub fn has_mkl() -> bool {
    
    todo!();
        /*
            return globalContext().hasMKL();
        */
}

#[inline] pub fn haslapack() -> bool {
    
    todo!();
        /*
            return globalContext().hasLAPACK();
        */
}

#[inline] pub fn hasmagma() -> bool {
    
    todo!();
        /*
            return globalContext().hasMAGMA();
        */
}

#[inline] pub fn hasmkldnn() -> bool {
    
    todo!();
        /*
            return globalContext().hasMKLDNN();
        */
}

#[inline] pub fn manual_seed(seed: u64)  {
    
    todo!();
        /*
            auto gen = globalContext().defaultGenerator(DeviceType::CPU);
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen.mutex());
        gen.set_current_seed(seed);
      }
      // NB: Sometimes we build with CUDA, but we don't have any GPUs
      // available. In that case, we must not seed CUDA; it will fail!
      const auto num_gpus = detail::getCUDAHooks().getNumGPUs();
      if (hasCUDA() && num_gpus > 0) {
        for (int i = 0; i < num_gpus; i++) {
          auto cuda_gen = globalContext().defaultGenerator(
            Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))
          );
          {
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(cuda_gen.mutex());
            cuda_gen.set_current_seed(seed);
          }
        }
      }
        */
}

/**
  | When the global flag `allow_tf32` is set to
  | true, cuBLAS handles are automatically
  | configured to use math mode
  | CUBLAS_TF32_TENSOR_OP_MATH.
  |
  | For some operators, such as addmv, TF32 offers
  | no performance improvement but causes precision
  | loss. To help this case, this class implements
  | a RAII guard that can be used to quickly
  | disable TF32 within its scope.
  |
  | Usage:
  |     NoTF32Guard disable_tf32;
  */
pub struct NoTF32Guard {
    changed: bool, // default = false
}

impl NoTF32Guard {
    
    pub fn should_disable_tf32() -> bool {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Context.cpp]

/**
  | TODO: This could be bad juju if someone
  | calls globalContext() in the destructor
  | of an object with static lifetime.
  |
  */
pub fn global_context<'a>() -> &'a mut Context {
    
    todo!();
        /*
            static Context globalContext_;
      return globalContext_;
        */
}

lazy_static!{
    /*
    static const char cublas_config_var_name[] = "CUBLAS_WORKSPACE_CONFIG";

    static const char* const cublas_deterministic_configs[] = { ":4096:8", ":16:8" };
    */
}

pub fn get_cpu_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return c10::GetCPUAllocator();
        */
}

/**
  | override_allow_tf32_flag = true
  |
  |    means the allow_tf32 flags are overrided and
  |    tf32 is force disabled
  |
  | override_allow_tf32_flag = false
  |
  |    means the original allow_tf32 flags are
  |    followed
  |
  */
lazy_static!{
    /*
    thread_local bool override_allow_tf32_flag = false;
    */
}

//-------------------------------
impl Drop for NoTF32Guard {
    fn drop(&mut self) {
        todo!();
        /*
            if (changed) {
        override_allow_tf32_flag = false;
      }
        */
    }
}

impl NoTF32Guard {
    
    pub fn new() -> Self {
    
        todo!();
        /*


            if (!override_allow_tf32_flag) {
        changed = true;
        override_allow_tf32_flag = true;
      }
        */
    }
    
    pub fn should_disable_tf32(&mut self) -> bool {
        
        todo!();
        /*
            return override_allow_tf32_flag;
        */
    }
}
