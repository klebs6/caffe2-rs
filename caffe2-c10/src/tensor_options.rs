crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/TensorOptions.h]

#[inline] pub fn dtype_or_default_scalar_type(dtype: Option<ScalarType>) -> ScalarType {
    
    todo!();
        /*
            return value_or_else(dtype, [] { return get_default_dtype_as_scalartype(); });
        */
}

#[inline] pub fn dtype_or_default_type_meta(dtype: Option<TypeMeta>) -> TypeMeta {
    
    todo!();
        /*
            return value_or_else(dtype, [] { return get_default_dtype(); });
        */
}

#[inline] pub fn layout_or_default(layout: Option<Layout>) -> Layout {
    
    todo!();
        /*
            return layout.value_or(kStrided);
        */
}

#[inline] pub fn device_or_default(device: Option<Device>) -> Device {
    
    todo!();
        /*
            return value_or_else(device, [] { return Device(kCPU); });
        */
}

#[inline] pub fn pinned_memory_or_default(pinned_memory: Option<bool>) -> bool {
    
    todo!();
        /*
            return pinned_memory.value_or(false);
        */
}

/**
  | A class to encapsulate construction axes of an
  | Tensor.  TensorOptions was designed to support
  | the Python style API for specifying
  | construction options on factory functions,
  | e.g.,
  |
  |     torch.zeros(2, 3, dtype=torch.int32)
  |
  | Because C++ doesn't natively support keyword
  | arguments, there must be another way of
  | specifying keyword-like arguments.
  | TensorOptions is a builder class which can be
  | used to construct this "dictionary" of keyword
  | arguments: functions which support
  | TensorOptions conventionally take this
  | argument optionally as their last argument.
  |
  | WARNING: In PyTorch, there are `Torch`
  | variants of factory functions, e.g.,
  | Torchzeros for zeros.  These return Variables
  | (while the stock ATen functions return plain
  | Tensors).  If you mix these functions up, you
  | WILL BE SAD.
  |
  | Rather than use the constructor of this class
  | directly, you should prefer to use the
  | constructor functions, and then chain setter
  | methods on top of them.
  |
  |     device(kCUDA).dtype(kInt)
  |     dtype(kInt)
  |
  | Additionally, anywhere a TensorOptions is
  | expected, you can directly pass kCUDA / kInt,
  | and it will implicitly convert to
  | a TensorOptions.
  |
  | Here are some recommended ways to create a 2x2
  | tensor of zeros with certain properties.
  | These all *implicitly* make use of
  | TensorOptions, even if they don't mention the
  | class explicitly:
  |
  |     zeros({2,2}, kCUDA);
  |     zeros({2,2}, kLong);
  |     zeros({2,2}, device(kCUDA).dtype(kLong()));
  |     zeros({2,2}, device({kCUDA, 1})); // place on device 1
  |     zeros({2,2}, requires_grad());
  |
  |
  | NOTE [ TensorOptions Constructors ]
  |
  | TensorOptions is like a dictionary with
  | entries from the set: {requires_grad, device,
  | dtype, layout}, where each entry may be
  | unspecified (i.e., is optional). It is used to
  | specify the properties of tensors in many
  | places both in C++ internal and API, e.g.,
  | tensor factory methods like `empty({10},
  | options)`, tensor conversions like
  | `tensor.to(...)`, etc.
  |
  | To provide a simple API that is consistent
  | with Python, where one can do
  |
  | `torch.empty(sizes, X)` with `X` being
  | a `torch.device`, `torch.dtype`, or a
  |
  | `torch.layout`, we want TensorOptions to be
  | implicitly convertible from
  |
  | `ScalarType dtype`, `Layout layout` and
  | `Device device`.
  |
  | Therefore, we have three implicit constructors
  | from each of these three types.
  |
  | This is sufficient for `ScalarType` and
  | `Layout` as they are simple Enum
  | classes. However, `Device` is an ordinary
  | class with implicit constructors
  |
  | `Device(DeviceType, DeviceIndex = -1)` and
  | `Device(string)` to be consistent with Python
  | API, where strings are treated as equivalent
  | with a
  |
  | `torch.device` object (e.g., "cuda:1" can be
  | passed to everywhere a
  |
  | `torch.device("cuda:1")` is accepted). To
  | support the syntax
  |
  | `empty({10}, {kCUDA, 1})` and
  | `tensor.to(kCUDA)`, we need to make sure that
  | `TensorOptions` is implicitly constructible
  | with any argments that a
  |
  | `Device` can constructed from. So we have,
  |
  |    /* implicit */ TensorOptions(T&& device) : TensorOptions() {
  |      this->set_device(device);
  |    }
  |
  |    template <typename... Args,
  |             typename = enable_if_t<is_constructible<Device,
  |             Args&&...>::value>>
  |    /* implicit */  TensorOptions(Args&&... args)
  |     : TensorOptions(Device(forward<Args>(args)...)) {}
  |
  |
  | But this will be problematic. Consider this:
  | `TensorOptions({kCUDA, 1})`. Compiler will
  | compain about ambiguity between the copy
  | constructor and the `Device` constructor
  | because `{kCUDA, 1}` can be converted to both
  | a `TensorOption` and a `Device`.
  |
  | To get around this, we templatize the `Device`
  | constructor. Since overload resolution is done
  | before template resolution, our problem is
  | solved.
  */
pub struct TensorOptions {

    // WARNING: If you edit TensorOptions to add
    // more options, you may need to adjust the
    // implementation of Tensor::options. The
    // criteria for whether or not Tensor::options
    // must be adjusted is whether or not the new
    // option you added should preserved by
    // functions such as empty_like(); if it should
    // be preserved, you must adjust options().
    //
    // TODO: MemoryFormat is not implemented in this
    // way

    // NB: We didn't use optional here, because then
    // we can't pack the has_***_ boolean fields.

    /**
      | 16-bit
      |
      */
    device:            Device, // default = kCPU

    /**
      | = TypeMeta::Make<float>(); // 16-bit
      |
      */
    dtype:             TypeMeta,

    /**
      | 8-bit
      |
      */
    layout:            Layout, // default = kStrided

    /**
      | = MemoryFormat::Contiguous; // 8-bit
      |
      */
    memory_format:     MemoryFormat,

    /**
      | Bitmask required here to get this to
      | fit inside 32 bits (or even 64 bits, for
      | that matter) : 1;
      |
      */
    requires_grad:     bool,

    /**
      | : 1;
      |
      */
    pinned_memory:     bool,

    /**
      | : 1;
      |
      */
    has_device:        bool,

    /**
      | : 1;
      |
      */
    has_dtype:         bool,

    /**
      | : 1;
      |
      */
    has_layout:        bool,

    /**
      | : 1;
      |
      */
    has_requires_grad: bool,

    /**
      | : 1;
      |
      */
    has_pinned_memory: bool,

    /**
      | : 1;
      |
      */
    has_memory_format: bool,
}

impl Default for TensorOptions {
    
    fn default() -> Self {
        todo!();
        /*


            : requires_grad_(false),
            pinned_memory_(false),
            has_device_(false),
            has_dtype_(false),
            has_layout_(false),
            has_requires_grad_(false),
            has_pinned_memory_(false),
            has_memory_format_(false)
        */
    }
}

impl TensorOptions {
    
    /// Constructs a `TensorOptions` object with
    /// the given layout.
    ///
    pub fn new_a(layout: Layout) -> Self {
    
        todo!();
        /*
        : tensor_options(),

            this->set_layout(layout);
        */
    }

    /// Constructs a `TensorOptions` object with the
    /// given device.
    ///
    /// See NOTE [ TensorOptions Constructors ] on
    /// why this is templatized.
    ///
    // template < typename T, typename = enable_if_t<is_same<decay_t<T>, Device>::value>>
    //
    pub fn new_b<T: PrimInt>(device: T) -> Self {
    
        todo!();
        /*
        : tensor_options(),

            this->set_device(forward<T>(device));
        */
    }

    /**
      | Constructs a `TensorOptions` object from
      |   arguments allowed in `Device` constructors.
      |
      | See NOTE [ TensorOptions Constructors ].
      |
      | NB: Ideally we only allow implicit
      |     constructors here. But there is no easy
      |     way to detect them. So we have this one
      |     that allows explicit constructors too.
      |
      | template < typename... Args, typename = enable_if_t<is_constructible<Device, Args&&...>::value>>
      */
    pub fn new_generic<Args>(args: Args) -> Self {
    
        todo!();
        /*


            : TensorOptions(Device(forward<Args>(args)...))
        */
    }

    /// Constructs a `TensorOptions` object with
    /// the given dtype.
    ///
    pub fn new_from_typemeta(dtype: TypeMeta) -> Self {
    
        todo!();
        /*
        : tensor_options(),

            this->set_dtype(dtype);
        */
    }

    /// legacy constructor to support ScalarType
    pub fn new_from_scalartype(dtype: ScalarType) -> Self {
    
        todo!();
        /*
        : tensor_options(),

            this->set_dtype(dtype);
        */
    }

    /// Constructs a `TensorOptions` object with
    /// the given memory format.
    ///
    pub fn new_from_memoryformat(memory_format: MemoryFormat) -> Self {
    
        todo!();
        /*
        : tensor_options(),

            set_memory_format(memory_format);
        */
    }

    /**
      | Return a copy of `TensorOptions` with
      | `device` set to the given one, or cleared if
      | `device` is `nullopt`.
      |
      */
    pub fn device_from_other(&self, device: Option<Device>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions r = *this;
        r.set_device(device);
        return r;
        */
    }

    /**
      | Return a copy of `TensorOptions` with
      | `device` set to the given one.
      |
      | (This overload ensures that variadic
      | template optional constructor for Device
      | work correctly.)
      |
      */
    pub fn device_from_args<Args>(&self, args: Args) -> TensorOptions {
    
        todo!();
        /*
            return device(
            optional<Device>(in_place, forward<Args>(args)...));
        */
    }

    /**
      | Return a copy of `TensorOptions` with
      | `dtype` set to the given one.
      |
      */
    pub fn dtype_copy_with_typemeta(&self, dtype: Option<TypeMeta>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions r = *this;
        r.set_dtype(dtype);
        return r;
        */
    }

    /// legacy function to support ScalarType
    ///
    pub fn dtype_with_scalartype(&self, dtype: Option<ScalarType>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions r = *this;
        r.set_dtype(dtype);
        return r;
        */
    }

    /// Since dtype is taken...
    pub fn dtype<T>(&mut self) -> &mut TensorOptions {
    
        todo!();
        /*
            dtype_ = TypeMeta::Make<T>();
        has_dtype_ = true;
        return *this;
        */
    }

    /// Sets the layout of the `TensorOptions`.
    pub fn layout_a(&self, layout: Option<Layout>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions r = *this;
        r.set_layout(layout);
        return r;
        */
    }

    /// Sets the `requires_grad` property of the
    /// `TensorOptions`.
    ///
    pub fn requires_grad_a(&self, requires_grad: Option<bool>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions r = *this;
        r.set_requires_grad(requires_grad);
        return r;
        */
    }

    /// Sets the `pinned_memory` property on the
    /// `TensorOptions`.
    ///
    pub fn pinned_memory_b(&self, pinned_memory: Option<bool>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions r = *this;
        r.set_pinned_memory(pinned_memory);
        return r;
        */
    }

    /// Sets the `memory_format` property on
    /// `TensorOptions`.
    ///
    pub fn memory_format(&self, memory_format: Option<MemoryFormat>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions r = *this;
        r.set_memory_format(memory_format);
        return r;
        */
    }

    /// Returns the device of the `TensorOptions`.
    ///
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return device_or_default(device_opt());
        */
    }

    /// Returns whether the device is specified.
    ///
    pub fn has_device(&self) -> bool {
        
        todo!();
        /*
            return has_device_;
        */
    }

    /**
      | Returns the device of the `TensorOptions`,
      | or `nullopt` if device is not specified.
      |
      */
    pub fn device_opt(&self) -> Option<Device> {
        
        todo!();
        /*
            return has_device_ ? make_optional(device_) : nullopt;
        */
    }

    /// Returns the device index of the
    /// `TensorOptions`.
    ///
    pub fn device_index(&self) -> i32 {
        
        todo!();
        /*
            return device().index();
        */
    }

    /// Returns the dtype of the `TensorOptions`.
    ///
    pub fn dtype_with_typemeta(&self) -> TypeMeta {
        
        todo!();
        /*
            return dtype_or_default(dtype_opt());
        */
    }

    /// Returns whether the dtype is specified.
    pub fn has_dtype(&self) -> bool {
        
        todo!();
        /*
            return has_dtype_;
        */
    }

    /**
      | Returns the dtype of the `TensorOptions`,
      | or `nullopt` if device is not specified.
      |
      */
    pub fn dtype_opt(&self) -> Option<TypeMeta> {
        
        todo!();
        /*
            return has_dtype_ ? make_optional(dtype_) : nullopt;
        */
    }

    /// Returns the layout of the `TensorOptions`.
    ///
    pub fn layout_b(&self) -> Layout {
        
        todo!();
        /*
            return layout_or_default(layout_opt());
        */
    }

    /// Returns whether the layout is specified.
    ///
    pub fn has_layout(&self) -> bool {
        
        todo!();
        /*
            return has_layout_;
        */
    }

    /**
      | Returns the layout of the `TensorOptions`,
      | or `nullopt` if layout is not specified.
      |
      */
    pub fn layout_opt(&self) -> Option<Layout> {
        
        todo!();
        /*
            return has_layout_ ? make_optional(layout_) : nullopt;
        */
    }

    /**
      | Returns the `requires_grad` property
      | of the `TensorOptions`.
      |
      */
    pub fn requires_grad_b(&self) -> bool {
        
        todo!();
        /*
            return has_requires_grad_ ? requires_grad_ : false;
        */
    }

    /// Returns whether the `requires_grad` is
    /// specified.
    ///
    pub fn has_requires_grad(&self) -> bool {
        
        todo!();
        /*
            return has_requires_grad_;
        */
    }

    /**
      | Returns the `requires_grad` property of the
      | `TensorOptions`, or `nullopt` if
      | `requires_grad` is not specified.
      |
      */
    pub fn requires_grad_opt(&self) -> Option<bool> {
        
        todo!();
        /*
            return has_requires_grad_ ? make_optional(requires_grad_)
                                  : nullopt;
        */
    }

    /// Returns the `pinned_memory` property of
    /// the `TensorOptions`.
    ///
    pub fn pinned_memory_a(&self) -> bool {
        
        todo!();
        /*
            return pinned_memory_or_default(pinned_memory_opt());
        */
    }

    /// Returns whether the `pinned_memory` is
    /// specified.
    ///
    pub fn has_pinned_memory(&self) -> bool {
        
        todo!();
        /*
            return has_pinned_memory_;
        */
    }

    /// Returns if the layout is sparse
    ///
    pub fn is_sparse(&self) -> bool {
        
        todo!();
        /*
            return layout_ == Layout::Sparse;
        */
    }
    
    pub fn is_sparse_csr(&self) -> bool {
        
        todo!();
        /*
            return layout_ == Layout::SparseCsr;
        */
    }

    /// For compatibility with legacy tensor.type()
    /// comparisons
    ///
    pub fn type_equal(&self, other: &TensorOptions) -> bool {
        
        todo!();
        /*
            return computeDispatchKey() == other.computeDispatchKey() &&
            typeMetaToScalarType(dtype_) == typeMetaToScalarType(other.dtype());
        */
    }

    /**
      | Returns the `pinned_memory` property of the
      | `TensorOptions`, or `nullopt` if
      | `pinned_memory` is not specified.
      |
      */
    pub fn pinned_memory_opt(&self) -> Option<bool> {
        
        todo!();
        /*
            return has_pinned_memory_ ? make_optional(pinned_memory_)
                                  : nullopt;
        */
    }

    /// Returns whether the `memory_layout` is
    /// specified
    ///
    pub fn has_memory_format(&self) -> bool {
        
        todo!();
        /*
            return has_memory_format_;
        */
    }

    // NB: memory_format() getter is PURPOSELY not
    // defined, as the default behavior of
    // memory_format varies from function to
    // function.

    /**
      | Returns the `memory_layout` property of
      | `TensorOptions, or `nullopt` if
      | `memory_format` is not specified.
      |
      */
    pub fn memory_format_opt(&self) -> Option<MemoryFormat> {
        
        todo!();
        /*
            return has_memory_format_ ? make_optional(memory_format_)
                                  : nullopt;
        */
    }

    /**
      | Resolves the ATen backend specified by the
      | current construction axes.
      |
      | TODO: Deprecate this
      */
    pub fn backend(&self) -> Backend {
        
        todo!();
        /*
            return dispatchKeyToBackend(computeDispatchKey());
        */
    }

    /**
      | Return the right-biased merge of two
      | TensorOptions.  This has the effect of
      | overwriting settings from self with
      | specified options of options.
      |
      | NB: This merging operation does NOT respect
      | device merges.
      |
      | For example, if you device({kCUDA, 1}).merge_in(kCUDA) 
      | you will get kCUDA in the end!  Functions
      | like Tensor.new_empty ensure the right
      | device is selected anyway by way of
      | a device guard.
      |
      */
    pub fn merge_in(&self, options: TensorOptions) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions merged = *this;
        if (options.has_device())
          merged.set_device(options.device_opt());
        if (options.has_dtype())
          merged.set_dtype(options.dtype_opt());
        if (options.has_layout())
          merged.set_layout(options.layout_opt());
        // NB: requires grad is right biased; not a logical AND/OR!
        if (options.has_requires_grad())
          merged.set_requires_grad(options.requires_grad_opt());
        if (options.has_pinned_memory())
          merged.set_pinned_memory(options.pinned_memory_opt());
        if (options.has_memory_format())
          merged.set_memory_format(options.memory_format_opt());
        return merged;
        */
    }

    /**
      | TODO remove after TensorOptions rationalization
      |
      */
    pub fn merge_memory_format(&self, optional_memory_format: Option<MemoryFormat>) -> TensorOptions {
        
        todo!();
        /*
            TensorOptions merged = *this;
        if (optional_memory_format.has_value()) {
          merged.set_memory_format(*optional_memory_format);
        }
        return merged;
        */
    }

    /**
      | INVARIANT: computeDispatchKey returns only
      | the subset of dispatch keys for which
      | dispatchKeyToBackend is injective, if it is
      | defined at all  (for the most part, this just
      | means that this function never returns an
      | Autograd key)
      |
      */
    pub fn compute_dispatch_key(&self) -> DispatchKey {
        
        todo!();
        /*
            return computeDispatchKey(
            optTypeMetaToScalarType(dtype_opt()), layout_opt(), device_opt());
        */
    }

    /*
      | These methods are currently private because
      | I'm not sure if it's wise to actually publish
      | them.  They are methods because I need them
      | in the constructor and the functional API
      | implementation.
      |
      | If you really, really need it, you can make
      | these public, but check if you couldn't just
      | do what you need with the functional API.
      | Similarly, these methods are not chainable,
      | because if you wanted chaining, you probably
      | want to use the functional API instead.
      | (It's probably OK to make these chainable,
      | because these functions are all explicitly
      | annotated with a ref-qualifier, the trailing
      | &, that makes them illegal to call on
      | temporaries.)
      */
 
    /// Mutably set the device of `TensorOptions`.
    ///
    pub fn set_device(&mut self, device: Option<Device>)  {
        
        todo!();
        /*
            if (device) {
          device_ = *device;
          has_device_ = true;
        } else {
          has_device_ = false;
        }
        */
    }

    /// Mutably set the dtype of `TensorOptions`.
    ///
    pub fn set_dtype(&mut self, dtype: Option<TypeMeta>)  {
        
        todo!();
        /*
            if (dtype) {
          dtype_ = *dtype;
          has_dtype_ = true;
        } else {
          has_dtype_ = false;
        }
        */
    }

    /// legacy function to support ScalarType
    ///
    pub fn set_dtype_a(&mut self, dtype: Option<ScalarType>)  {
        
        todo!();
        /*
            if (dtype) {
          dtype_ = scalarTypeToTypeMeta(*dtype);
          has_dtype_ = true;
        } else {
          has_dtype_ = false;
        }
        */
    }

    /// Mutably set the layout of `TensorOptions`.
    ///
    pub fn set_layout(&mut self, layout: Option<Layout>)  {
        
        todo!();
        /*
            if (layout) {
          layout_ = *layout;
          has_layout_ = true;
        } else {
          has_layout_ = false;
        }
        */
    }

    /// Mutably set the `requires_grad` property
    /// of `TensorOptions`.
    ///
    pub fn set_requires_grad(&mut self, requires_grad: Option<bool>)  {
        
        todo!();
        /*
            if (requires_grad) {
          requires_grad_ = *requires_grad;
          has_requires_grad_ = true;
        } else {
          has_requires_grad_ = false;
        }
        */
    }

    /// Mutably set the `pinned_memory` property
    /// of `TensorOptions`.
    ///
    pub fn set_pinned_memory(&mut self, pinned_memory: Option<bool>)  {
        
        todo!();
        /*
            if (pinned_memory) {
          pinned_memory_ = *pinned_memory;
          has_pinned_memory_ = true;
        } else {
          has_pinned_memory_ = false;
        }
        */
    }

    /**
      | Mutably set the `memory_Format` property
      | of `TensorOptions`.
      |
      */
    pub fn set_memory_format(&mut self, memory_format: Option<MemoryFormat>)  {
        
        todo!();
        /*
            if (memory_format) {
          memory_format_ = *memory_format;
          has_memory_format_ = true;
        } else {
          has_memory_format_ = false;
        }
        */
    }
}

/**
  | We should aspire to fit in one machine-size
  | word; but a size greater than two words is too
  | much.  (We are doing terribly on 32-bit archs,
  | where we require three machine size words to
  | store tensor options.  Eek!)
  |
  | TensorOptions must fit in 128-bits"
  |
  */
const_assert!(
    size_of::<TensorOptions>() <= size_of::<i64>() * 2
);

/**
  | Convenience function that returns
  | a `TensorOptions` object with the `dtype`
  | set to the given one.
  |
  */
#[inline] pub fn dtype_with_typemeta(dtype: TypeMeta) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions().dtype(dtype);
        */
}

/**
  | legacy function to support ScalarType
  |
  */
#[inline] pub fn dtype_with_scalartype(dtype: ScalarType) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions().dtype(scalarTypeToTypeMeta(dtype));
        */
}

/**
  | Convenience function that returns
  | a `TensorOptions` object with the `layout`
  | set to the given one.
  |
  */
#[inline] pub fn layout(layout: Layout) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions().layout(layout);
        */
}

/**
  | Convenience function that returns
  | a `TensorOptions` object with the `device` set
  | to the given one.
  |
  */
#[inline] pub fn device(device: Device) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions().device(move(device));
        */
}

/**
  | Convenience function that returns
  | a `TensorOptions` object with the `device` set
  | to Cuda and the `device_index` set to the
  | given one.
  |
  */
#[inline] pub fn device_index(device_index: i16) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions().device_index(device_index);
        */
}

/**
  | Convenience function that returns
  | a `TensorOptions` object with the
  | `requires_grad` set to the given one.
  |
  */
#[inline] pub fn requires_grad(requires_grad: Option<bool>) -> TensorOptions {

    let requires_grad: bool = requires_grad.unwrap_or(true);

    todo!();
        /*
            return TensorOptions().requires_grad(requires_grad);
        */
}

/**
  | Convenience function that returns
  | a `TensorOptions` object with the
  | `memory_format` set to the given one.
  |
  */
#[inline] pub fn memory_format(memory_format: MemoryFormat) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions().memory_format(memory_format);
        */
}

#[inline] pub fn dtype<T>() -> TensorOptions {

    todo!();
        /*
            return dtype(TypeMeta::Make<T>());
        */
}

#[inline] pub fn to_string(options: TensorOptions) -> String {
    
    todo!();
        /*
            ostringstream stream;
      stream << options;
      return stream.str();
        */
}

/**
  | This is intended to be a centralized
  | location by which we can determine what
  | an appropriate DispatchKey for a tensor is.
  |
  */
#[inline] pub fn compute_dispatch_key(
        dtype:  Option<ScalarType>,
        layout: Option<Layout>,
        device: Option<Device>) -> DispatchKey {
    
    todo!();
        /*
            const auto layout_ = layout_or_default(layout);
      const auto device_ = device_or_default(device);
      switch (layout_) {
        case Layout::Strided: {
          const auto dtype_ = dtype_or_default(dtype);
          switch (device_.type()) {
            case DeviceType::CPU: {
              if (isQIntType(dtype_)) {
                return DispatchKey::QuantizedCPU;
              }
              return DispatchKey::CPU;
            }
            case DeviceType::CUDA: {
              if (isQIntType(dtype_)) {
                return DispatchKey::QuantizedCUDA;
              }
              return DispatchKey::Cuda;
            }
            case DeviceType::XPU: {
              if (isQIntType(dtype_)) {
                return DispatchKey::QuantizedXPU;
              }
              return DispatchKey::XPU;
            }
            case DeviceType::MKLDNN:
            case DeviceType::OPENGL:
            case DeviceType::OPENCL:
            case DeviceType::IDEEP:
              TORCH_INTERNAL_ASSERT(
                  0,
                  "This is a grandfathered Caffe2 device type ",
                  device_.type(),
                  ", it shouldn't ever convert to a DispatchKey.  File a bug describing what you were doing if you think this is in error.");
            case DeviceType::HIP:
              return DispatchKey::HIP;
            case DeviceType::FPGA:
              return DispatchKey::FPGA;
            case DeviceType::MSNPU:
              return DispatchKey::MSNPU;
            case DeviceType::XLA:
              return DispatchKey::XLA;
            case DeviceType::MLC:
              return DispatchKey::MLC;
            case DeviceType::Vulkan:
              return DispatchKey::Vulkan;
            case DeviceType::Metal:
              return DispatchKey::Metal;
            case DeviceType::Meta:
              return DispatchKey::Meta;
            case DeviceType::HPU:
              return DispatchKey::HPU;
            default:
              TORCH_CHECK_NOT_IMPLEMENTED(
                  false,
                  "Unsupported device type for dense layout: ",
                  device_.type());
          }
        }
        case Layout::Sparse:
          switch (device_.type()) {
            case DeviceType::CPU:
              return DispatchKey::SparseCPU;
            case DeviceType::CUDA:
              return DispatchKey::SparseCUDA;
            case DeviceType::HIP:
              return DispatchKey::SparseHIP;
            case DeviceType::XPU:
              return DispatchKey::SparseXPU;
            default:
              TORCH_CHECK_NOT_IMPLEMENTED(
                  false,
                  "Unsupported device type for sparse layout: ",
                  device_.type());
          }
        case Layout::Mkldnn:
          switch (device_.type()) {
            case DeviceType::CPU:
              return DispatchKey::MkldnnCPU;
            default:
              TORCH_CHECK_NOT_IMPLEMENTED(
                  false,
                  "Unsupported device type for mkldnn layout: ",
                  device_.type());
          }
        case Layout::SparseCsr:
          switch (device_.type()) {
            case DeviceType::CPU:
              return DispatchKey::SparseCsrCPU;
            case DeviceType::CUDA:
              return DispatchKey::SparseCsrCUDA;
            default:
              AT_ERROR(
                  "Unsupported device type for sparse CSR layout: ",
                  device_.type());
          }
        default:
          TORCH_CHECK(false, "Unsupported layout: ", layout_);
      }
        */
}



#[inline] pub fn dispatch_key_to_layout(dispatch_key: DispatchKey) -> Layout {
    
    todo!();
        /*
            switch (dispatch_key) {
        case DispatchKey::SparseCPU:
        case DispatchKey::SparseCUDA:
        case DispatchKey::SparseHIP:
        case DispatchKey::SparseXPU:
        case DispatchKey::SparseCsrCPU:
        case DispatchKey::SparseCsrCUDA:
          return Layout::Sparse;
        case DispatchKey::MkldnnCPU:
          return Layout::Mkldnn;
        default:
          return Layout::Strided;
      }
        */
}

#[inline] pub fn dispatch_key_to_device_type(dispatch_key: DispatchKey) -> DeviceType {
    
    todo!();
        /*
            switch (dispatch_key) {
        // stuff that's real
        case DispatchKey::CPU:
        case DispatchKey::SparseCPU:
        case DispatchKey::MkldnnCPU:
        case DispatchKey::QuantizedCPU:
        case DispatchKey::AutogradCPU:
          return DeviceType::CPU;
        case DispatchKey::Cuda:
        case DispatchKey::SparseCUDA:
        case DispatchKey::QuantizedCUDA:
        case DispatchKey::AutogradCUDA:
          return DeviceType::CUDA;
        case DispatchKey::HIP:
        case DispatchKey::SparseHIP:
          return DeviceType::HIP;
        case DispatchKey::XLA:
        case DispatchKey::AutogradXLA:
          return DeviceType::XLA;
        case DispatchKey::Vulkan:
          return DeviceType::Vulkan;
        case DispatchKey::Meta:
          return DeviceType::Meta;

        // stuff that people are actively developing
        case DispatchKey::XPU:
        case DispatchKey::SparseXPU:
        case DispatchKey::QuantizedXPU:
        case DispatchKey::AutogradXPU:
          return DeviceType::XPU;
        case DispatchKey::MLC:
        case DispatchKey::AutogradMLC:
          return DeviceType::MLC;
        case DispatchKey::HPU:
        case DispatchKey::AutogradHPU:
          return DeviceType::HPU;

        // stuff that isn't real
        case DispatchKey::MSNPU:
          return DeviceType::MSNPU;
        default:
          TORCH_CHECK(
              false,
              "DispatchKey ",
              dispatch_key,
              " doesn't correspond to a device");
      }
        */
}

#[inline] pub fn dispatch_key_to_tensor_options(dispatch_key: DispatchKey) -> TensorOptions {
    
    todo!();
        /*
            return TensorOptions()
          .layout(dispatchKeyToLayout(dispatch_key))
          .device(dispatchKeyToDeviceType(dispatch_key));
        */
}

//-------------------------------------------[.cpp/pytorch/c10/core/TensorOptions.cpp]

impl fmt::Display for TensorOptions {
    
    /**
      | Note: TensorOptions properties are all
      | optional, but (almost) all have getters that
      | supply a default when the corresponding
      | property is missing.
      |
      | Here we print the values returned by the
      | default-supplying getters for properties that
      | have them, along with an annotation if the
      | value is returned by default. This gives the
      | full picture of both the object's internal
      | state and what its getters will return.
      */
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            auto print = [&](const char* label, auto prop, bool has_prop) {
        stream << label << boolalpha << prop << (has_prop ? "" : " (default)");
      };

      print("TensorOptions(dtype=", options.dtype(), options.has_dtype());
      print(", device=", options.device(), options.has_device());
      print(", layout=", options.layout(), options.has_layout());
      print(
          ", requires_grad=", options.requires_grad(), options.has_requires_grad());
      print(
          ", pinned_memory=", options.pinned_memory(), options.has_pinned_memory());

      // note: default-supplying memory_format() getter not provided; no canonical
      // default
      stream << ", memory_format=";
      if (options.has_memory_format()) {
        stream << *options.memory_format_opt();
      } else {
        stream << "(nullopt)";
      }
      stream << ")";

      return stream;
        */
    }
}
