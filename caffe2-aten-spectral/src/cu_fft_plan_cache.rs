crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/CuFFTPlanCache.h]

/**
  | Enum representing the FFT type
  |
  */
#[repr(i8)]
pub enum CuFFTTransformType {

    /// Complex-to-complex
    C2C,  

    /// Real-to-complex
    R2C,  

    /// Complex-to-real
    C2R,  
}

/**
  | This struct is used to let us easily compute
  | hashes of the parameters.
  |
  | It will be the **key** to the plan cache.
  |
  */
#[derive(Default)]
pub struct CuFFTParams {

    /**
      | between 1 and max_rank, i.e., 1 <= signal_ndim
      | <= 3
      |
      */
    signal_ndim:    i64,

    /**
      | These include additional batch dimension
      | as well.
      |
      */
    sizes:          [i64; max_rank + 1],

    input_strides:  [i64; max_rank + 1],
    output_strides: [i64; max_rank + 1],
    fft_type:       CuFFTTransformType,
    value_type:     ScalarType,
}

impl CuFFTParams {

    pub fn new(
        in_strides:   &[i32],
        out_strides:  &[i32],
        signal_sizes: &[i32],
        fft_type:     CuFFTTransformType,
        value_type:   ScalarType) -> Self {
    
        todo!();
        /*


            // Padding bits must be zeroed for hashing
        memset(this, 0, sizeof(*this));
        signal_ndim_ = signal_sizes.size() - 1;
        fft_type_ = fft_type;
        value_type_ = value_type;

        TORCH_INTERNAL_ASSERT(in_strides.size() == signal_sizes.size());
        TORCH_INTERNAL_ASSERT(out_strides.size() == signal_sizes.size());
        TORCH_INTERNAL_ASSERT(1 <= signal_ndim_ && signal_ndim_ <= max_rank);

        copy(signal_sizes.cbegin(), signal_sizes.cend(), sizes_);
        copy(in_strides.cbegin(), in_strides.cend(), input_strides_);
        copy(out_strides.cbegin(), out_strides.cend(), output_strides_);
        */
    }
}

const_assert!(is_trivial::<CuFFTParams>());

/**
  | Returns true if the transform type has
  | complex input
  |
  */
#[inline] pub fn cufft_complex_input(ty: CuFFTTransformType) -> bool {
    
    todo!();
        /*
            switch (type) {
        case CuFFTTransformType::C2C:
        case CuFFTTransformType::C2R:
          return true;

        case CuFFTTransformType::R2C:
          return false;
      }
      TORCH_INTERNAL_ASSERT(false);
        */
}

/**
  | Returns true if the transform type has
  | complex output
  |
  */
#[inline] pub fn cufft_complex_output(ty: CuFFTTransformType) -> bool {
    
    todo!();
        /*
            switch (type) {
        case CuFFTTransformType::C2C:
        case CuFFTTransformType::R2C:
          return true;

        case CuFFTTransformType::C2R:
          return false;
      }
      TORCH_INTERNAL_ASSERT(false);
        */
}

/**
  | Create transform type enum from bools
  | representing if input and output are
  | complex
  |
  */
#[inline] pub fn get_cu_fft_transform_type(
        complex_input:  bool,
        complex_output: bool) -> CuFFTTransformType {
    
    todo!();
        /*
            if (complex_input && complex_output) {
        return CuFFTTransformType::C2C;
      } else if (complex_input && !complex_output) {
        return CuFFTTransformType::C2R;
      } else if (!complex_input && complex_output) {
        return CuFFTTransformType::R2C;
      }
      TORCH_INTERNAL_ASSERT(false, "Real to real FFTs are not supported");
        */
}

pub struct CuFFTHandle {
    handle: CuFFTHandle,
}

impl Default for CuFFTHandle {
    
    fn default() -> Self {
        todo!();
        /*


            CUFFT_CHECK(cufftCreate(&handle_));
        */
    }
}

impl CuFFTHandle {

    pub fn get(&mut self) -> &mut CuFFTHandle {
        
        todo!();
        /*
            return handle_;
        */
    }
    
    pub fn get(&self) -> &CuFFTHandle {
        
        todo!();
        /*
            return handle_;
        */
    }
}

impl Drop for CuFFTHandle {
    fn drop(&mut self) {
        todo!();
        /*
            // Not using fftDestroy() for rocFFT to work around double freeing of handles
    #ifndef __HIP_PLATFORM_HCC__
        cufftDestroy(handle_);
    #endif
        */
    }
}

#[inline(always)] pub fn is_pow_of_two(x: i64) -> bool {
    
    todo!();
        /*
            return (x & (x - 1)) == 0;
        */
}

#[cfg(__HIP_PLATFORM_HCC__)]
pub type cufft_Sizeype = i32;

#[cfg(not(__HIP_PLATFORM_HCC__))]
pub type cufft_Sizeype = i64;

pub type CuFFTDimVector = SmallVector<CuFFTSizeType,kDimVectorStaticSize>;

/**
  | Struct representing a tensor in CuFFT's data
  | layout for planning transforms
  |
  | See NOTE [ cuFFT Embedded Strides ].
  |
  */
pub struct CuFFTDataLayout {
    embed:      CuFFTDimVector,
    stride:     CuFFTSizeType,
    dist:       CuFFTSizeType,
    must_clone: bool,
    simple:     bool,
}

/**
  | Returns a cufft embedding for a contiguous
  | signal of the given size.
  |
  | e.g. if the input is cloned, this will be the
  | resulting data layout
  |
  | See NOTE [ cuFFT Embedded Strides ].
  */
#[inline] pub fn cufft_simple_embed(
        sizes:    &[i32],
        onesided: bool) -> CuFFTDataLayout {
    
    todo!();
        /*
            CuFFTDataLayout layout;
      layout.simple = true;
      layout.must_clone = false;
      layout.embed.assign(sizes.cbegin() + 1, sizes.cend());
      if (onesided) {
        layout.embed.back() = sizes.back() / 2 + 1;
      }
      layout.stride = 1;
      layout.dist = 1;
      for (const auto& len : layout.embed) {
        layout.dist *= len;
      }
      return layout;
        */
}

/**
  | Convert strides to a CuFFT embedded
  | representation.
  |
  | If strides cannot be embedded, returns a simple
  | layout and sets must_clone flag
  |
  | See NOTE [ cuFFT Embedded Strides ].
  |
  */
#[inline] pub fn as_cufft_embed(
        strides:  &[i32],
        sizes:    &[i32],
        onesided: bool) -> CuFFTDataLayout {
    
    todo!();
        /*
            const auto signal_ndim = strides.size() - 1;
      CuFFTDataLayout layout;
      auto last_stride = strides[signal_ndim];
      layout.must_clone = (last_stride <= 0);

      const auto last_dim_size = onesided ?
          sizes[signal_ndim] / 2 + 1 : sizes[signal_ndim];
      const auto signal_numel = multiply_integers(sizes.slice(1, sizes.size() - 2)) * last_dim_size;

      // Zero stides are not allowed, even if the batch size is one.
      // If that happens just set a dummy case
      if (sizes[0] == 1) {
        layout.dist = signal_numel;
      } else if (strides[0] == 0) {
        layout.must_clone = true;
      } else {
        layout.dist = strides[0];
      }

      // Calculate the embedding shape, or set must_clone if the strides cannot be embedded
      layout.embed.resize(signal_ndim);
      for (auto i = signal_ndim - 1; !layout.must_clone && i > 0; i--) {
        auto stride = strides[i];
        if (sizes[i] == 1) {
          layout.embed[i] = 1;
        } else if (stride > 0 && stride % last_stride == 0) {
          layout.embed[i] = stride / last_stride;
          last_stride = stride;
        } else {
          layout.must_clone = true;
        }
      }

      if (layout.must_clone) {
        // If the input needs to be cloned, assume it will be contiguous
        layout = cufft_simple_embed(sizes, onesided);
        layout.must_clone = true;
      } else {
        layout.embed[0] = sizes[1];
        layout.stride = strides[signal_ndim];
        // Determine if layout represents a simple embedding (contiguous data)
        layout.simple = [&] {
          for (i64 i = 1; i < signal_ndim - 1; ++i) {
            if (layout.embed[i] != sizes[i + 1]) {
              return false;
            }
          }

          return (layout.stride == 1 && layout.dist == signal_numel &&
              layout.embed.back() == last_dim_size);
        }();
      }
      return layout;
        */
}

/**
  | This class contains all the information needed
  | to execute a cuFFT plan:
  |
  |   1. the plan
  |   2. whether to clone input before executing the plan
  |   3. the workspace size needed
  |
  | This class will be the **value** in the plan
  | cache.
  |
  | It **owns** the raw plan via a unique_ptr.
  |
  | Only move semantics is enought for this
  | class. Although we already use unique_ptr for
  | the plan, still remove copy constructor and
  | assignment op so we don't accidentally copy
  | and take perf hit.
  */
pub struct CuFFTConfig {
    plan_ptr:    CuFFTHandle,
    clone_input: bool,
    ws_size:     i64,
    fft_type:    CuFFTTransformType,
    value_type:  ScalarType,
}

impl CuFFTConfig {

    pub fn new(params: &CuFFTParams) -> Self {
    
        todo!();
        /*


            :
          CuFFTConfig(
              IntArrayRef(params.input_strides_, params.signal_ndim_ + 1),
              IntArrayRef(params.output_strides_, params.signal_ndim_ + 1),
              IntArrayRef(params.sizes_, params.signal_ndim_ + 1),
              params.fft_type_,
              params.value_type_)
        */
    }

    /**
      | For complex types, strides are in units of
      | 2 * element_size(dtype)
      |
      | sizes are for the full signal, including
      | batch size and always two-sided
      |
      */
    pub fn new(
        in_strides:  &[i32],
        out_strides: &[i32],
        sizes:       &[i32],
        fft_type:    CuFFTTransformType,
        dtype:       ScalarType) -> Self {
    
        todo!();
        /*


            : fft_type_(fft_type), value_type_(dtype) 

        // signal sizes (excluding batch dim)
        CuFFTDimVector signal_sizes(sizes.begin() + 1, sizes.end());

        // input batch size
        const i64 batch = sizes[0];
        const i64 signal_ndim = sizes.size() - 1;

        // Since cuFFT has limited non-unit stride support and various constraints, we
        // use a flag to keep track throughout this function to see if we need to
        // input = input.clone();

    #ifdef __HIP_PLATFORM_HCC__
        // clone input to avoid issues with hipfft clobering the input and failing tests
        clone_input = true;
    #else
        clone_input = false;
    #endif

        // For half, base strides on the real part of real-to-complex and
        // complex-to-real transforms are not supported. Since our output is always
        // contiguous, only need to check real-to-complex case.
        if (dtype == ScalarType::Half) {
          // cuFFT on half requires compute capability of at least SM_53
          auto dev_prop = getCurrentDeviceProperties();
          TORCH_CHECK(dev_prop->major >= 5 && !(dev_prop->major == 5 && dev_prop->minor < 3),
                   "cuFFT doesn't support signals of half type with compute "
                   "capability less than SM_53, but the device containing input half "
                   "tensor only has SM_", dev_prop->major, dev_prop->minor);
          for (i64 i = 0; i < signal_ndim; i++) {
            TORCH_CHECK(is_pow_of_two(sizes[i + 1]),
                "cuFFT only supports dimensions whose sizes are powers of two when"
                " computing in half precision, but got a signal size of",
                sizes.slice(1));
          }
          clone_input |= in_strides.back() != 1;
        }

        CuFFTDataLayout in_layout;
        if (clone_input) {
          in_layout = cufft_simple_embed(sizes, fft_type == CuFFTTransformType::C2R);
        } else {
          in_layout = as_cufft_embed(in_strides, sizes, fft_type == CuFFTTransformType::C2R);
        }
        auto out_layout = as_cufft_embed(out_strides, sizes, fft_type == CuFFTTransformType::R2C);
        TORCH_INTERNAL_ASSERT(!out_layout.must_clone, "Out strides cannot be represented as CuFFT embedding");
        clone_input |= in_layout.must_clone;

        // Check if we can take advantage of simple data layout.
        //
        // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.

        const bool simple_layout = in_layout.simple && out_layout.simple;

    #ifdef __HIP_PLATFORM_HCC__
        hipfftType exec_type = [&]{
          if (dtype == kFloat) {
            switch (fft_type) {
              case CuFFTTransformType::C2C: return HIPFFT_C2C;
              case CuFFTTransformType::R2C: return HIPFFT_R2C;
              case CuFFTTransformType::C2R: return HIPFFT_C2R;
            }
          } else if (dtype == kDouble) {
            switch (fft_type) {
              case CuFFTTransformType::C2C: return HIPFFT_Z2Z;
              case CuFFTTransformType::R2C: return HIPFFT_D2Z;
              case CuFFTTransformType::C2R: return HIPFFT_Z2D;
            }
          }
          TORCH_CHECK(false, "hipFFT doesn't support transforms of type: ", dtype);
        }();
    #else
        cudaDataType itype, otype, exec_type;
        const auto complex_input = cufft_complex_input(fft_type);
        const auto complex_output = cufft_complex_output(fft_type);
        if (dtype == ScalarType::Float) {
          itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
          otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
          exec_type = CUDA_C_32F;
        } else if (dtype == ScalarType::Double) {
          itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
          otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
          exec_type = CUDA_C_64F;
        } else if (dtype == ScalarType::Half) {
          itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
          otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
          exec_type = CUDA_C_16F;
        } else {
          TORCH_CHECK(false, "cuFFT doesn't support tensor of type: ", dtype);
        }
    #endif

        // disable auto allocation of workspace to use THC allocator
        CUFFT_CHECK(cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));

        usize ws_Size;

        // make plan
        if (simple_layout) {
          // If with unit-stride, we tell cuFFT by setting inembed == onembed == NULL.
          // In such case, cuFFT ignores istride, ostride, idist, and odist
          // by assuming istride = ostride = 1.
          //
          // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.
    #ifdef __HIP_PLATFORM_HCC__
          CUFFT_CHECK(hipfftMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
            /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1,
            /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1,
            exec_type, batch, &ws_Size));
    #else
          CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
            /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
            /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
            batch, &ws_Size, exec_type));
    #endif
        } else {
    #ifdef __HIP_PLATFORM_HCC__
          CUFFT_CHECK(hipfftMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
            in_layout.embed.data(), in_layout.stride, in_layout.dist,
            out_layout.embed.data(), out_layout.stride, out_layout.dist,
            exec_type, batch, &ws_Size));
    #else
          CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim, signal_sizes.data(),
                in_layout.embed.data(), in_layout.stride, in_layout.dist, itype,
                out_layout.embed.data(), out_layout.stride, out_layout.dist, otype,
                batch, &ws_Size, exec_type));
    #endif
        }
        ws_size = static_cast<i64>(ws_Size);
        */
    }
    
    pub fn plan(&self) -> &CuFFTHandle {
        
        todo!();
        /*
            return plan_ptr.get();
        */
    }
    
    pub fn transform_type(&self) -> CuFFTTransformType {
        
        todo!();
        /*
            return fft_type_;
        */
    }
    
    pub fn data_type(&self) -> ScalarType {
        
        todo!();
        /*
            return value_type_;
        */
    }
    
    pub fn should_clone_input(&self) -> bool {
        
        todo!();
        /*
            return clone_input;
        */
    }
    
    pub fn workspace_size(&self) -> i64 {
        
        todo!();
        /*
            return ws_size;
        */
    }
}

/**
  | -----------
  | @note
  | 
  | the max plan number for CUDA version
  | < 10 has to be 1023 due to a bug that fails
  | on the 1024th plan
  |
  */
#[cfg(CUDA_VERSION_LT_10000)]
pub const CUFFT_MAX_PLAN_NUM: usize = 1023;

#[cfg(CUDA_VERSION_LT_10000)]
pub const CUFFT_DEFAULT_CACHE_SIZE: usize = CUFFT_MAX_PLAN_NUM;

#[cfg(not(CUDA_VERSION_LT_10000))]
pub const CUFFT_MAX_PLAN_NUM: usize = usize::MAX;

/**
  | The default max cache size chosen for CUDA
  | version > 10 is arbitrary.
  |
  | This number puts a limit on how big of a plan
  | cache should we maintain by default. Users
  | can always configure it via
  | cufft_set_plan_cache_max_size.
  */
#[cfg(not(CUDA_VERSION_LT_10000))]
pub const CUFFT_DEFAULT_CACHE_SIZE: usize = 4096;

static_assert!(CUFFT_MAX_PLAN_NUM >= 0 && CUFFT_MAX_PLAN_NUM <= usize::max, "CUFFT_MAX_PLAN_NUM not in usize range");
static_assert!(CUFFT_DEFAULT_CACHE_SIZE >= 0 && CUFFT_DEFAULT_CACHE_SIZE <= CUFFT_MAX_PLAN_NUM, "CUFFT_DEFAULT_CACHE_SIZE not in [0, CUFFT_MAX_PLAN_NUM] range");

/**
  | This cache assumes that the mapping
  | from key to value never changes.
  | 
  | This is **NOT** thread-safe. Please
  | use a mutex when using it **AND** the
  | value returned from try_emplace_value.
  | 
  | The contract of using this cache is that
  | try_emplace_value should only be used
  | when the max_size is positive.
  |
  */
pub struct CuFFTParamsLRUCache {
    mutex:      RawMutex,
    usage_list: LinkedList<Kv>,
    cache_map:  Map,
    max_size:   usize,
}

pub mod cu_fft_params_lru_cache {

    use super::*;

    lazy_static!{
        /*
        using kv_t = typename pair<CuFFTParams, CuFFTConfig>;
          using map_t = typename unordered_map<reference_wrapper<CuFFTParams>,
                                                    typename list<kv_t>::iterator,
                                                    ParamsHash<CuFFTParams>,
                                                    ParamsEqual<CuFFTParams>>;
          using map_kkv_iter_t = typename map_t::iterator;
        */
    }
}

impl Default for CuFFTParamsLRUCache {
    
    fn default() -> Self {
        todo!();
        /*
        : cu_fft_params_lru_cache(CUFFT_DEFAULT_CACHE_SIZE),

        
        */
    }
}

impl CuFFTParamsLRUCache {
    
    pub fn new(max_size: i64) -> Self {
    
        todo!();
        /*


            _set_max_size(max_size);
        */
    }
    
    pub fn new(other: CuFFTParamsLRUCache) -> Self {
    
        todo!();
        /*


            :
        _usage_list(move(other._usage_list)),
        _cache_map(move(other._cache_map)),
        _max_size(other._max_size)
        */
    }
    
    pub fn assign_from(&mut self, other: CuFFTParamsLRUCache) -> &mut CuFFTParamsLRUCache {
        
        todo!();
        /*
            _usage_list = move(other._usage_list);
        _cache_map = move(other._cache_map);
        _max_size = other._max_size;
        return *this;
        */
    }

    /**
      | If key is in this cache, return the cached
      | config. Otherwise, emplace the config in this
      | cache and return it.
      |
      | Return const reference because CuFFTConfig
      | shouldn't be tampered with once created.
      |
      */
    pub fn lookup(&mut self, params: CuFFTParams) -> &CuFFTConfig {
        
        todo!();
        /*
            AT_ASSERT(_max_size > 0);

        map_kkv_iter_t map_it = _cache_map.find(params);
        // Hit, put to list front
        if (map_it != _cache_map.end()) {
          _usage_list.splice(_usage_list.begin(), _usage_list, map_it->second);
          return map_it->second->second;
        }

        // Miss
        // remove if needed
        if (_usage_list.size() >= _max_size) {
          auto last = _usage_list.end();
          last--;
          _cache_map.erase(last->first);
          _usage_list.pop_back();
        }

        // construct new plan at list front, then insert into _cache_map
        _usage_list.emplace_front(piecewise_construct,
                           forward_as_tuple(params),
                           forward_as_tuple(params));
        auto kv_it = _usage_list.begin();
        _cache_map.emplace(piecewise_construct,
                    forward_as_tuple(kv_it->first),
                    forward_as_tuple(kv_it));
        return kv_it->second;
        */
    }
    
    pub fn clear(&mut self)  {
        
        todo!();
        /*
            _cache_map.clear();
        _usage_list.clear();
        */
    }
    
    pub fn resize(&mut self, new_size: i64)  {
        
        todo!();
        /*
            _set_max_size(new_size);
        auto cur_size = _usage_list.size();
        if (cur_size > _max_size) {
          auto delete_it = _usage_list.end();
          for (usize i = 0; i < cur_size - _max_size; i++) {
            delete_it--;
            _cache_map.erase(delete_it->first);
          }
          _usage_list.erase(delete_it, _usage_list.end());
        }
        */
    }
    
    pub fn size(&self) -> usize {
        
        todo!();
        /*
            return _cache_map.size();
        */
    }
    
    pub fn max_size(&self) -> usize {
        
        todo!();
        /*
            return _max_size;
        */
    }

    /**
      | Only sets size and does value check.
      | Does not resize the data structures.
      |
      */
    pub fn set_max_size(&mut self, new_size: i64)  {
        
        todo!();
        /*
            // We check that 0 <= new_size <= CUFFT_MAX_PLAN_NUM here. Since
        // CUFFT_MAX_PLAN_NUM is of type usize, we need to do non-negativity check
        // first.
        TORCH_CHECK(new_size >= 0,
                 "cuFFT plan cache size must be non-negative, but got ", new_size);
        TORCH_CHECK(new_size <= CUFFT_MAX_PLAN_NUM,
                 "cuFFT plan cache size can not be larger than ", CUFFT_MAX_PLAN_NUM, ", but got ", new_size);
        _max_size = static_cast<usize>(new_size);
        */
    }
}

/**
  | Since ATen is separated into CPU build and CUDA
  | build, we need a way to call these functions
  | only when CUDA is loaded. We use CUDA hooks for
  | this purpose (at cuda/detail/CUDAHooks.cpp),
  | and call the hooked functions from the actual
  | native function counterparts (at
  | native/SpectralOps.cpp), i.e.,
  |
  | _cufft_get_plan_cache_max_size, _cufft_set_plan_cache_max_size
  | _cufft_get_plan_cache_size, and _cufft_clear_plan_cache.
  */
pub fn cufft_get_plan_cache_max_size_impl(device_index: i64) -> i64 {
    
    todo!();
        /*
        
        */
}

pub fn cufft_set_plan_cache_max_size_impl(
        device_index: i64,
        max_size:     i64)  {
    
    todo!();
        /*
        
        */
}

pub fn cufft_get_plan_cache_size_impl(device_index: i64) -> i64 {
    
    todo!();
        /*
        
        */
}

pub fn cufft_clear_plan_cache_impl(device_index: i64)  {
    
    todo!();
        /*
        
        */
}
