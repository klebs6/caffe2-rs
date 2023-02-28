crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/Functions.h]

/**
  | These functions are defined in ATen/Utils.cpp.
  |
  */
#[macro_export] macro_rules! tensor {
    ($T:ident, $S:ident) => {
        /*
        
           Tensor tensor(ArrayRef<T> values, const TensorOptions& options); 
          inline Tensor tensor(                                                       
              initializer_list<T> values, const TensorOptions& options) {        
            return tensor(ArrayRef<T>(values), options);                          
          }                                                                           
          inline Tensor tensor(T value, const TensorOptions& options) {               
            return tensor(ArrayRef<T>(value), options);                           
          }                                                                           
          inline Tensor tensor(ArrayRef<T> values) {                                  
            return tensor(move(values), dtype(k##S));                    
          }                                                                           
          inline Tensor tensor(initializer_list<T> values) {                     
            return tensor(ArrayRef<T>(values));                                   
          }                                                                           
          inline Tensor tensor(T value) {                                             
            return tensor(ArrayRef<T>(value));                                    
          }
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types_and3!{Bool, Half, BFloat16, TENSOR}
    at_forall_complex_types!{TENSOR}
    */
}


/**
  | Provides a fluent API to construct tensors
  | from external data.
  |
  | The fluent API can be used instead of
  | `from_blob` functions in case the required set
  | of parameters does not align with the existing
  | overloads.
  |
  |     Tensor tensor = for_blob(data, sizes)
  |             .strides(strides)
  |             .context(context, [](void *ctx) { delete static_cast<Ctx*>(ctx); })
  |             .options(...)
  |             .make_tensor();
  |
  */
pub struct TensorMaker {
    data:    *mut c_void,
    sizes:   &[i32],

    /**
      | {};
      |
      */
    strides: Option<&[i32]>,


    /**
      | {};
      |
      */
    deleter: fn(_0: *mut c_void) -> (),


    /**
      | {nullptr, noopDelete};
      |
      */
    ctx:     Box<c_void,ContextDeleter>,


    /**
      | {};
      |
      */
    device:  Option<Device>,


    /**
      | {};
      |
      */
    opts:    TensorOptions,
}

impl TensorMaker {

    pub type ContextDeleter = DeleterFnPtr;
    
    pub fn strides(&mut self, value: Option<&[i32]>) -> &mut TensorMaker {
        
        todo!();
        /*
            strides_ = value;

        return *this;
        */
    }
    
    pub fn deleter(&mut self, value: fn(_0: *mut c_void) -> ()) -> &mut TensorMaker {
        
        todo!();
        /*
            deleter_ = move(value);

        return *this;
        */
    }
    
    pub fn context(&mut self, 
        value:   *mut c_void,
        deleter: ContextDeleter) -> &mut TensorMaker {
        let deleter: ContextDeleter = deleter.unwrap_or(nullptr);

        todo!();
        /*
            ctx_ = unique_ptr<void, ContextDeleter>{
            value, deleter != nullptr ? deleter : noopDelete};

        return *this;
        */
    }
    
    pub fn target_device(&mut self, value: Option<Device>) -> &mut TensorMaker {
        
        todo!();
        /*
            device_ = value;

        return *this;
        */
    }
    
    pub fn options(&mut self, value: TensorOptions) -> &mut TensorMaker {
        
        todo!();
        /*
            opts_ = value;

        return *this;
        */
    }
    
    pub fn make_tensor(&mut self) -> Tensor {
        
        todo!();
        /*
        
        */
    }
    
    pub fn new(
        data:  *mut c_void,
        sizes: &[i32]) -> Self {
    
        todo!();
        /*


            : data_{data}, sizes_{sizes}
        */
    }
    
    pub fn compute_storage_size(&self) -> usize {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_data_ptr_from_deleter(&self) -> DataPtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_data_ptr_from_context(&mut self) -> DataPtr {
        
        todo!();
        /*
        
        */
    }
    
    pub fn make_temp_sizes(&self) -> &[i32] {
        
        todo!();
        /*
        
        */
    }
}

#[inline] pub fn for_blob(
        data:  *mut c_void,
        sizes: &[i32]) -> TensorMaker {
    
    todo!();
        /*
            return TensorMaker{data, sizes};
        */
}

#[inline] pub fn from_blob_a(
        data:          *mut c_void,
        sizes:         &[i32],
        strides:       &[i32],
        deleter:       &fn(_0: *mut c_void) -> (),
        options:       Option<&TensorOptions>,
        target_device: Option<Device>) -> Tensor {

    let target_device: Option<Device> = target_device.unwrap_or(nullopt);

    todo!();
        /*
            return for_blob(data, sizes)
          .strides(strides)
          .deleter(deleter)
          .options(options)
          .target_device(target_device)
          .make_tensor();
        */
}

#[inline] pub fn from_blob_b(
        data:    *mut c_void,
        sizes:   &[i32],
        deleter: &fn(_0: *mut c_void) -> (),
        options: &TensorOptions) -> Tensor {
    let options: &TensorOptions = options.unwrap_or(default);

    todo!();
        /*
            return for_blob(data, sizes)
          .deleter(deleter)
          .options(options)
          .make_tensor();
        */
}

#[inline] pub fn from_blob_c(
        data:    *mut c_void,
        sizes:   &[i32],
        strides: &[i32],
        options: &TensorOptions) -> Tensor {
    let options: &TensorOptions = options.unwrap_or(default);

    todo!();
        /*
            return for_blob(data, sizes)
          .strides(strides)
          .options(options)
          .make_tensor();
        */
}

#[inline] pub fn from_blob_d(
        data:    *mut c_void,
        sizes:   &[i32],
        options: &TensorOptions) -> Tensor {
    let options: &TensorOptions = options.unwrap_or(default);

    todo!();
        /*
            return for_blob(data, sizes).options(options).make_tensor();
        */
}



#[inline] pub fn numel(tensor: &Tensor) -> i64 {
    
    todo!();
        /*
            return tensor.numel();
        */
}


#[inline] pub fn size(
        tensor: &Tensor,
        dim:    i64) -> i64 {
    
    todo!();
        /*
            return tensor.size(dim);
        */
}


#[inline] pub fn stride(
        tensor: &Tensor,
        dim:    i64) -> i64 {
    
    todo!();
        /*
            return tensor.stride(dim);
        */
}


#[inline] pub fn is_complex(tensor: &Tensor) -> bool {
    
    todo!();
        /*
            return tensor.is_complex();
        */
}


#[inline] pub fn is_floating_point(tensor: &Tensor) -> bool {
    
    todo!();
        /*
            return tensor.is_floating_point();
        */
}


#[inline] pub fn is_signed(tensor: &Tensor) -> bool {
    
    todo!();
        /*
            return tensor.is_signed();
        */
}


#[inline] pub fn is_inference(tensor: &Tensor) -> bool {
    
    todo!();
        /*
            return tensor.is_inference();
        */
}


#[inline] pub fn is_conj(tensor: &Tensor) -> bool {
    
    todo!();
        /*
            return tensor.is_conj();
        */
}


#[inline] pub fn conj(tensor: &Tensor) -> Tensor {
    
    todo!();
        /*
            return tensor.conj();
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/Functions.cpp]

/**
  | Special C++ only overloads for std()-like
  | functions (See gh-40287)
  |
  | These are needed because int -> bool conversion
  | takes precedence over int -> IntArrayRef
  |
  | So, for example std(0) would select the
  | std(unbiased=False) overload
  */
pub fn var(
        self_: &Tensor,
        dim:   i32) -> Tensor {
    
    todo!();
        /*
            return var(self, IntArrayRef{dim});
        */
}


pub fn var_mean(
        self_: &Tensor,
        dim:   i32) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return var_mean(self, IntArrayRef{dim});
        */
}


pub fn std(
        self_: &Tensor,
        dim:   i32) -> Tensor {
    
    todo!();
        /*
            return std(self, IntArrayRef{dim});
        */
}


pub fn std_mean(
        self_: &Tensor,
        dim:   i32) -> (Tensor,Tensor) {
    
    todo!();
        /*
            return std_mean(self, IntArrayRef{dim});
        */
}

/**
  | Special C++ only overloads for convnd functions
  | (See gh-45667)
  |
  | These are needed because {1, 2} is ambiguous
  | between string and IntArrayRef overloads
  |
  */
pub fn conv1d(
        input:    &Tensor,
        weight:   &Tensor,
        bias:     &Tensor,
        stride:   &[i32],
        padding:  InitializerList<i64>,
        dilation: &[i32],
        groups:   i64) -> Tensor {

    let dilation: &[i32] = dilation.unwrap_or(1);
    let groups: i64 = groups.unwrap_or(1);
    
    todo!();
        /*
            auto padding = IntArrayRef(padding_);
      return conv1d(input, weight, bias, stride, padding, dilation, groups);
        */
}


pub fn conv2d(
        input:    &Tensor,
        weight:   &Tensor,
        bias:     &Tensor,
        stride:   &[i32],
        padding:  InitializerList<i64>,
        dilation: &[i32],
        groups:   i64) -> Tensor {

    let dilation: &[i32] = dilation.unwrap_or(1);
    let groups: i64 = groups.unwrap_or(1);
    
    todo!();
        /*
            auto padding = IntArrayRef(padding_);
      return conv2d(input, weight, bias, stride, padding, dilation, groups);
        */
}


pub fn conv3d(
    input:    &Tensor,
    weight:   &Tensor,
    bias:     &Tensor,
    stride:   &[i32],
    padding:  InitializerList<i64>,
    dilation: &[i32],
    groups:   i64) -> Tensor {

    let dilation: &[i32] = dilation.unwrap_or(1);
    let groups: i64 = groups.unwrap_or(1);
    
    todo!();
        /*
            auto padding = IntArrayRef(padding_);
      return conv3d(input, weight, bias, stride, padding, dilation, groups);
        */
}

pub fn noop_delete(_0: *mut c_void)  {
    
    todo!();
        /*
        
        */
}

impl TensorMaker {
    
    pub fn make_tensor(&mut self) -> Tensor {
        
        todo!();
        /*
            AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
      tracer::NoTracerDispatchMode tracer_guard{};

      check_size_nonnegative(sizes_);

      TORCH_CHECK_VALUE(
          !deleter_ || !ctx_,
          "The deleter and context arguments are mutually exclusive.");

      if (device_ == nullopt) {
        device_ = globalContext().getDeviceFromPtr(data_, opts_.device().type());
      }

      if (opts_.device().has_index()) {
        // clang-format off
        TORCH_CHECK_VALUE(
            opts_.device() == *device_,
            "Specified device ", opts_.device(), " does not match device of data ", *device_);
        // clang-format on
      }

      usize size_bytes = computeStorageSize();

      DataPtr data_ptr{};
      if (deleter_) {
        data_ptr = makeDataPtrFromDeleter();
      } else {
        data_ptr = makeDataPtrFromContext();
      }

      Storage storage{Storage::use_byte_Size{}, size_bytes, move(data_ptr)};

      Tensor tensor = make_tensor<TensorImpl>(
          move(storage), opts_.computeDispatchKey(), opts_.dtype());

      if (sizes_.size() != 1 || sizes_[0] != 0) {
        TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

        if (strides_) {
          tensor_impl->set_sizes_and_strides(sizes_, *strides_);
        } else {
          tensor_impl->set_sizes_contiguous(sizes_);
        }
      }

      return tensor;
        */
    }
    
    pub fn compute_storage_size(&self) -> usize {
        
        todo!();
        /*
            usize itemsize = opts_.dtype().itemsize();

      if (strides_) {
        return computeStorageNbytes(sizes_, *strides_, itemsize);
      }

      usize size = 1;
      for (i64 s : sizes_) {
        size *= static_cast<usize>(s);
      }
      return size * itemsize;
        */
    }
    
    #[inline] pub fn make_data_ptr_from_deleter(&self) -> DataPtr {
        
        todo!();
        /*
            return InefficientStdFunctionContext::makeDataPtr(data_, deleter_, *device_);
        */
    }
    
    #[inline] pub fn make_data_ptr_from_context(&mut self) -> DataPtr {
        
        todo!();
        /*
            return DataPtr{data_, ctx_.release(), ctx_.get_deleter(), *device_};
        */
    }
    
    pub fn make_temp_sizes(&self) -> &[i32] {
        
        todo!();
        /*
            static i64 zeros[5] = {0, 0, 0, 0, 0};
      if (opts_.has_memory_format()) {
        MemoryFormat format = *opts_.memory_format_opt();
        if (format == MemoryFormat::ChannelsLast) {
          return IntArrayRef(zeros, 4);
        }
        if (format == MemoryFormat::ChannelsLast3d) {
          return IntArrayRef(zeros, 5);
        }
      }
      return IntArrayRef(zeros, 1);
        */
    }
}
