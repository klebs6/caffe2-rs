crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Utils.h]

#[macro_export] macro_rules! at_disallow_copy_and_assign {
    ($TypeName:ident) => {
        /*
        
          TypeName(const TypeName&) = delete; 
          void operator=(const TypeName&) = delete
        */
    }
}

lazy_static!{
    /*
    int _crash_if_asan(int);
    */
}

/**
  | TODO: This unwrapping code is ONLY used for TH
  | bindings; once TH goes away, we can delete this
  | function
  |
  */
#[inline] pub fn checked_dense_tensor_unwrap(
    expr:        &Tensor,
    name:        *const u8,
    pos:         i32,
    api:         *const u8,
    allow_null:  bool,
    device_type: DeviceType,
    scalar_type: ScalarType) -> *mut TensorImpl {

    todo!();
        /*
            if(allowNull && !expr.defined()) {
        return nullptr;
      }
      if (expr.layout() != Layout::Strided) {
        AT_ERROR("Expected dense tensor but got ", expr.layout(),
                 " for argument #", pos, " '", name, "' in call to ", api);
      }
      if (expr.device().type() != device_type) {
        AT_ERROR("Expected object of device type ", device_type, " but got device type ", expr.device().type(),
                 " for argument #", pos, " '", name, "' in call to ", api);
      }
      if (expr.scalar_type() != scalar_type) {
        AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr.scalar_type(),
                 " for argument #", pos, " '", name, "' in call to ", api);
      }
      return expr.unsafeGetTensorImpl();
        */
}

/**
  | Converts a &[Tensor] (i.e. &[Tensor] to
  | vector of TensorImpl*)
  |
  | NB: This is ONLY used by legacy TH bindings,
  | and ONLY used by cat.
  |
  | Once cat is ported entirely to ATen this can be
  | deleted!
  |
  */
#[inline] pub fn checked_dense_tensor_list_unwrap(
    tensors:     &[Tensor],
    name:        *const u8,
    pos:         i32,
    device_type: DeviceType,
    scalar_type: ScalarType) -> Vec<*mut TensorImpl> {
    
    todo!();
        /*
            vector<TensorImpl*> unwrapped;
      unwrapped.reserve(tensors.size());
      for (const auto i : irange(tensors.size())) {
        const auto& expr = tensors[i];
        if (expr.layout() != Layout::Strided) {
          AT_ERROR("Expected dense tensor but got ", expr.layout(),
                   " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
        }
        if (expr.device().type() != device_type) {
          AT_ERROR("Expected object of device type ", device_type, " but got device type ", expr.device().type(),
                   " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
        }
        if (expr.scalar_type() != scalar_type) {
          AT_ERROR("Expected object of scalar type ", scalar_type, " but got scalar type ", expr.scalar_type(),
                   " for sequence element ", i , " in sequence argument at position #", pos, " '", name, "'");
        }
        unwrapped.emplace_back(expr.unsafeGetTensorImpl());
      }
      return unwrapped;
        */
}

pub fn check_intlist<const N: usize>(
    list: &[i64],
    name: *const u8,
    pos:  i32) -> Array<i64,N> {

    todo!();
        /*
            if (list.empty()) {
        // TODO: is this necessary?  We used to treat nullptr-vs-not in IntList differently
        // with strides as a way of faking optional.
        list = {};
      }
      auto res = array<i64, N>();
      if (list.size() == 1 && N > 1) {
        res.fill(list[0]);
        return res;
      }
      if (list.size() != N) {
        AT_ERROR("Expected a list of ", N, " ints but got ", list.size(), " for argument #", pos, " '", name, "'");
      }
      copy_n(list.begin(), N, res.begin());
      return res;
        */
}

/**
  | Utility function to static cast input
  | Generator* to the backend generator
  | type (CPU/CUDAGeneratorImpl etc.)
  |
  */
#[inline] pub fn check_generator<T>(gen: Option<dyn GeneratorInterface>) -> *mut T {

    todo!();
        /*
            TORCH_CHECK(gen.has_value(), "Expected Generator but received nullopt");
      TORCH_CHECK(gen->defined(), "Generator with undefined implementation is not allowed");
      TORCH_CHECK(T::device_type() == gen->device().type(), "Expected a '", T::device_type(), "' device type for generator but found '", gen->device().type(), "'");
      return gen->get<T>();
        */
}

/**
  | Utility function used in tensor implementations,
  | which supplies the default generator
  | to tensors, if an input generator is
  | not supplied. 
  |
  | The input Generator*
  | is also static casted to the backend
  | generator type (CPU/CUDAGeneratorImpl
  | etc.)
  |
  */
#[inline] pub fn get_generator_or_default<T>(
    gen:         &Option<dyn GeneratorInterface>,
    default_gen: &dyn GeneratorInterface

) -> *mut T {

    todo!();
        /*
            return gen.has_value() && gen->defined() ? check_generator<T>(gen) : check_generator<T>(default_gen);
        */
}

#[inline] pub fn check_size_nonnegative(size: &[i32])  {
    
    todo!();
        /*
            for (auto x: size) {
        TORCH_CHECK(x >= 0, "Trying to create tensor with negative dimension ", x, ": ", size);
      }
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/Utils.cpp]

pub fn crash_if_asan(arg: i32) -> i32 {
    
    todo!();
        /*
      volatile char x[3];
      x[arg] = 0;
      return x[0];
        */
}

/**
  | empty_cpu is used in ScalarOps.h, which can be
  | referenced by other ATen files.
  |
  | Since we want to decouple direct referencing
  | native symbols and only access native symbols
  | through dispatching, we move its implementation
  | here.
  |
  */
pub fn empty_cpu(
    size:              &[i32],
    dtype_opt:         Option<ScalarType>,
    layout_opt:        Option<Layout>,
    device_opt:        Option<Device>,
    pin_memory_opt:    Option<bool>,
    memory_format_opt: Option<MemoryFormat>) -> Tensor {

    todo!();
        /*
            auto device = device_or_default(device_opt);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType_CPU);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

      bool pin_memory = pinned_memory_or_default(pin_memory_opt);
      Allocator* allocator;
      if (pin_memory) {
        allocator = getCUDAHooks().getPinnedMemoryAllocator();
      } else {
        allocator = getCPUAllocator();
      }
      auto dtype = dtype_or_default(dtype_opt);

      return empty_generic(size, allocator, DispatchKey::CPU, dtype, device, memory_format_opt);
        */
}

pub fn empty_generic(
    size:              &[i32],
    allocator:         *mut Allocator,

    // technically this can be inferred from the device, but usually the
    // correct setting is obvious from the call site so just make callers
    // pass it in
    dispatch_key:      DispatchKey,
    scalar_type:       ScalarType,
    device:            Device,
    memory_format_opt: Option<MemoryFormat>) -> Tensor {

    todo!();
        /*
            check_size_nonnegative(size);

      i64 nelements = multiply_integers(size);
      TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
      i64 size_bytes = nelements * dtype.itemsize();
      auto storage_impl = make_intrusive<StorageImpl>(
          StorageImpl::use_byte_size_t(),
          size_bytes,
          allocator->allocate(size_bytes),
          allocator,
          /*resizeable=*/true);

      auto tensor = make_tensor<TensorImpl>(
          move(storage_impl), dispatch_key, dtype);
      // Default TensorImpl has size [0]
      if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
      }

      if (memory_format_opt.has_value()) {
        // Restriding a just-created empty contiguous tensor does nothing.
        if (*memory_format_opt != MemoryFormat::Contiguous) {
          tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
        }
      }

      return tensor;
        */
}

pub fn tensor_cpu<T>(
        values:  &[T],
        options: &TensorOptions) -> Tensor {

    todo!();
        /*
            auto result = empty(values.size(), options);
      AT_ASSERT(result.is_contiguous());
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "tensor_cpu", [&] {
        copy(
            values.begin(), values.end(), result.template data_ptr<Scalar>());
      });
      return result;
        */
}

pub fn tensor_backend<T>(
        values:  &[T],
        options: &TensorOptions) -> Tensor {

    todo!();
        /*
            auto cpu_tensor = tensor_cpu(values, options.device(DeviceType_CPU));
      return cpu_tensor.to(options.device());
        */
}

pub fn tensor_complex_cpu<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            auto result = empty(values.size(), options);
      AT_ASSERT(result.is_contiguous());
      AT_DISPATCH_COMPLEX_TYPES(result.scalar_type(), "tensor_cpu", [&] {
        copy(
            values.begin(), values.end(), result.template data_ptr<Scalar>());
      });
      return result;
        */
}

pub fn tensor_complex_backend<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            auto cpu_tensor = tensor_complex_cpu(values, options.device(DeviceType_CPU));
      return cpu_tensor.to(options.device());
        */
}

lazy_static!{
    /*
    macro_rules! tensor {
        ($T:ident, $_1:ident) => {
            /*
            
              Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { 
                if (options.device().type() != DeviceType_CPU) {          
                  return tensor_backend(values, options);           
                } else {                                                        
                  return tensor_cpu(values, options);               
                }                                                               
              }
            */
        }
    }

    at_forall_scalar_types_and3!{Bool, Half, BFloat16, TENSOR}
    */
}

lazy_static!{
    /*
    macro_rules! tensor {
        ($T:ident, $_1:ident) => {
            /*
            
              Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { 
                if (options.device().type() != DeviceType_CPU) {          
                  return tensor_complex_backend(values, options);   
                } else {                                                        
                  return tensor_complex_cpu(values, options);       
                }                                                               
              }
            */
        }
    }

    at_forall_complex_types!{tensor}
    */
}

