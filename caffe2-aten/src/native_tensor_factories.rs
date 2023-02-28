// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorFactories.h]

/**
  | Different combinations of row, col, and offset
  | can lead to two cases:
  |
  | Case 1 - Trapezoid (Triangle as a special case): row + offset <= col
  |    Example A: offset > 0
  |      1 1 0 0 0
  |      1 1 1 0 0
  |      1 1 1 1 0
  |    Example B: offset <= 0
  |      0 0 0
  |      1 0 0
  |      1 1 0
  |    In this case, we calculate the number of elements in the first row and
  |    last row of the tril respectively, and then compute the tril size.
  |
  | Case 2 - Trapezoid + Rectangle: row + offset > col
  |    Example:
  |      1 1 0
  |      1 1 1
  |      1 1 1
  |    In this case, we first calculate the size of top trapezoid, and then
  |    calculate the size of the bottom rectangle.
  */
#[inline] pub fn get_tril_size(
        row:    i64,
        col:    i64,
        offset: i64) -> i64 {
    
    todo!();
        /*
            // number of elements in the first row of the tril
      auto m_first_row = offset > 0 ?
        min<i64>(col, 1 + offset) : // upper bounded by col
        row + offset > 0; // either 0 or 1
      // number of elements in the last row of the tril, bounded by [0, col]
      auto m_last_row = max<i64>(0, min<i64>(col, row + offset));
      // number of rows, bounded by [0, row]
      auto n_row_all = max<i64>(0, min<i64>(row, row + offset));
      auto n_row_trapezoid = (m_last_row - m_first_row + 1);

      // calculate # of elements in the top trapezoid
      auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

      // calculate # of elements in the bottom rectangle if there is any
      auto diff_row = n_row_all - n_row_trapezoid;
      if (diff_row > 0) {
        tril_size += diff_row * col;
      }

      return tril_size;
        */
}

#[inline] pub fn check_args(
        row:        i64,
        col:        i64,
        layout_opt: Option<Layout>)  {
    
    todo!();
        /*
            TORCH_CHECK(row >= 0, "row must be non-negative, got", row);
      TORCH_CHECK(col >= 0, "col must be non-negative, got", col);
      if (layout_opt.has_value()) {
        TORCH_CHECK(
          *layout_opt == kStrided,
          "only support layout=torch.strided, got",
          *layout_opt)
      }
        */
}

/**
  | assumes maximum value in created tensor
  | is n-1 (e.g., torch.randperm(n))
  |
  */
#[inline] pub fn check_supported_max_int_with_precision(
        n:      i64,
        tensor: &Tensor)  {
    
    todo!();
        /*
            // match defined() to behavior of checks below
      TORCH_CHECK(scalar_tensor(n>0?n-1:n, tensor.options()).defined(),
                  "n is too large for result tensor type: '", tensor.toString(), "'");

      // Ensure sufficient precision for floating point representation.
      switch (tensor.scalar_type()) {
        case ScalarType::Half:
          TORCH_CHECK(n <= (i64(1) << 11) + 1, "n cannot be greater than 2049 for Half type.");
          break;
        case ScalarType::Float:
          TORCH_CHECK(n <= (i64(1) << 24) + 1, "n cannot be greater than 2^24+1 for Float type.");
          break;
        case ScalarType::Double:  // Unlikely to happen, but doesn't hurt to check
          TORCH_CHECK(n <= (i64(1) << 53) + 1, "n cannot be greater than 2^53+1 for Double type.");
          break;
        default:
          break;
      }
        */
}

pub type BinaryFn = fn(_0: &mut TensorIterator) -> ();

declare_dispatch!{binary_fn, complex_stub}
declare_dispatch!{binary_fn, polar_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorFactories.cpp]

pub fn window_function_checks(
    function_name: *const u8,
    options:       &TensorOptions,
    window_length: i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
          options.layout() != kSparse,
          function_name,
          " is not implemented for sparse types, got: ",
          options);
      TORCH_CHECK(
          isFloatingType(typeMetaToScalarType(options.dtype())) || isComplexType(typeMetaToScalarType(options.dtype())),
          function_name,
          " expects floating point dtypes, got: ",
          options);
      TORCH_CHECK(
          window_length >= 0,
          function_name,
          " requires non-negative window_length, got window_length=",
          window_length);
        */
}

define_dispatch!{complex_stub}
define_dispatch!{polar_stub}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ arange ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn arange_a(
        end:        &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::arange(/*start=*/0, end, dtype, layout, device, pin_memory);
        */
}

pub fn arange_b(
        start:      &Scalar,
        end:        &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::arange(
          start, end, /*step=*/1, dtype, layout, device, pin_memory);
        */
}

pub fn arange_c(
        start:      &Scalar,
        end:        &Scalar,
        step:       &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      bool set_to_integral_dtype = !options.has_dtype() &&
           // bool inputs are considered integral
           start.isIntegral(true) &&
           end.isIntegral(true) &&
           step.isIntegral(true);

      Tensor result = set_to_integral_dtype
          ? empty({0}, options.dtype(ScalarType::Long))
          : empty({0}, options);
      return arange_out(result, start, end, step);
        */
}

pub fn arange_out_a(
        end:    &Scalar,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return arange_out(result, /*start=*/0, end);
        */
}

pub fn arange_out_b(
        result: &mut Tensor,
        start:  &Scalar,
        end:    &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return arange_out(result, start, end, /*step=*/1);
        */
}

pub fn dim_arange(
        like: &Tensor,
        dim:  i64) -> Tensor {
    
    todo!();
        /*
            return arange(like.size(dim), like.options().dtype(kLong));
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complex / polar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn complex_check_floating(
        a: &Tensor,
        b: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK((a.scalar_type() == kFloat || a.scalar_type() == kDouble) &&
                  (b.scalar_type() == kFloat || b.scalar_type() == kDouble),
                  "Expected both inputs to be Float or Double tensors but got ",
                  a.scalar_type(), " and ", b.scalar_type());
        */
}

pub fn complex_check_dtype(
        result: &Tensor,
        a:      &Tensor,
        b:      &Tensor)  {
    
    todo!();
        /*
            complex_check_floating(a, b);
      TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                  "Expected object of scalar type ", a.scalar_type(),
                  " but got scalar type ", b.scalar_type(), " for second argument");
      TORCH_CHECK(result.scalar_type() == toComplexType(a.scalar_type()),
                  "Expected object of scalar type ", toComplexType(a.scalar_type()),
                  " but got scalar type ", result.scalar_type(),
                  " for argument 'out'");
        */
}

pub fn complex_out(
        real:   &Tensor,
        imag:   &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            complex_check_dtype(result, real, imag);
      auto iter = TensorIteratorConfig()
          .add_output(result)
          .add_input(real)
          .add_input(imag)
          .check_all_same_dtype(false)
          .build();
      complex_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn complex(
        real: &Tensor,
        imag: &Tensor) -> Tensor {
    
    todo!();
        /*
            complex_check_floating(real, imag);
      TensorOptions options = real.options();
      options = options.dtype(toComplexType(real.scalar_type()));
      Tensor result = empty(0, options);
      return complex_out(result, real, imag);
        */
}

pub fn polar_out(
        abs:    &Tensor,
        angle:  &Tensor,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            complex_check_dtype(result, abs, angle);
      auto iter = TensorIteratorConfig()
          .add_output(result)
          .add_input(abs)
          .add_input(angle)
          .check_all_same_dtype(false)
          .build();
      polar_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn polar(
        abs:   &Tensor,
        angle: &Tensor) -> Tensor {
    
    todo!();
        /*
            complex_check_floating(abs, angle);
      TensorOptions options = abs.options();
      options = options.dtype(toComplexType(abs.scalar_type()));
      Tensor result = empty(0, options);
      return polar_out(result, abs, angle);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn empty_cpu(
        size:              &[i32],
        dtype_opt:         Option<ScalarType>,
        layout_opt:        Option<Layout>,
        device_opt:        Option<Device>,
        pin_memory_opt:    Option<bool>,
        memory_format_opt: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            return empty_cpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
        */
}

pub fn empty(
        size:                   &[i32],
        names:                  Option<DimnameList>,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      if (!names.has_value()) {
        return empty(size, options, optional_memory_format);
      }
      TORCH_CHECK(options.layout() == Layout::Strided,
          "NYI: named tensors only support strided layout");
      TORCH_CHECK(options.device().is_cpu() || options.device().is_cuda(),
          "NYI: named tensors only support CPU and CUDA tensors");
      auto result = empty(size, options, optional_memory_format);
      internal_set_names_inplace(result, names);
      return result;
        */
}

pub fn empty_strided_cpu(
        size:           &[i32],
        stride:         &[i32],
        dtype_opt:      Option<ScalarType>,
        layout_opt:     Option<Layout>,
        device_opt:     Option<Device>,
        pin_memory_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            check_size_nonnegative(size);
      auto t = native::empty_cpu({0}, dtype_opt, layout_opt, device_opt, pin_memory_opt);
      native::resize_impl_cpu_(t.unsafeGetTensorImpl(), size, stride);
      return t;
        */
}

pub fn empty_out(
        size:                   &[i32],
        optional_memory_format: Option<MemoryFormat>,
        result:                 &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // Preferably, this argument would not be accepted by _out, but the code
      // generator requires the out and non-out overloads to match exactly
      TORCH_CHECK(
          !optional_memory_format.has_value(),
          "'memory_format' argument is incompatible with 'out' tensor argument");
      check_size_nonnegative(size);
      if (result.is_sparse()) {
        result.sparse_resize_and_clear_(size, size.size(), 0);
      } else {
        result.resize_(size);
      }
      return result;
        */
}

/**
  | Temporary type cast operators. These are needed
  | to trace type-casts now since Type's are not
  | supported in the IR. Instead, we call down to
  | these specialized operators for each datatype.
  |
  | TODO: remove when we have Type support in the
  | IR
  */
#[macro_export] macro_rules! define_cast_op {
    ($_1:ident, $n:ident) => {
        /*
        
          Tensor _cast_##n(const Tensor& self, bool non_blocking) {      
            if (self.scalar_type() == ScalarType::n)                     
              return self;                                               
            return self.to(ScalarType::n, non_blocking);                 
          }
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types_and3!{Bool, Half, BFloat16, DEFINE_CAST_OP}
    */
}

pub fn empty_like(
        self_:                  &Tensor,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(
        !(options_.has_memory_format() && optional_memory_format.has_value()),
        "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
        "the redundant setter.");

      TensorOptions options =
          self.options()
              .merge_in(options_)
              .merge_memory_format(optional_memory_format);

      TORCH_CHECK(
          !(options.layout() != kStrided &&
              optional_memory_format.has_value()),
          "memory format option is only supported by strided tensors");
      if (options.layout() == kSparse && self.is_sparse()) {
        auto result = empty({0}, options); // to be resized
        result.sparse_resize_and_clear_(
            self.sizes(), self.sparse_dim(), self.dense_dim());
        return result;
      }

      auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);

      if (self.is_quantized()) {

        // TODO: To support all features of MemoryFormat::Preserve we need to add
        // _empty_affine_quantized_strided function and use it similarly to
        // Tensor clone(const Tensor& src, optional<MemoryFormat> optional_memory_format)
        // if (self.is_non_overlapping_and_dense()) -> _empty_affine_quantized_strided
        if (memory_format == MemoryFormat::Preserve) {
          memory_format = self.suggest_memory_format();
        }

        // Note [Explicit nullopt MemoryFormat argument]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Some functions which we call default the OPTIONAL MemoryFormat
        // argument to something that's not nullopt.  If we pass the
        // MemoryFormat via TensorOptions, we must explicitly disable this
        // defaulting process, by explicitly passing nullopt for the MemoryFormat
        // argument.  When codegen is adjusted so we can delete this argument from
        // the method signature, the argument will just disappear entirely.
        //
        // BTW, there are a few places where the optional MemoryFormat is None,
        // but I still pass in nullopt for robustness.

        // We could check if dtype is still quantized?  But then should we shift/scale
        // the q_zero_point / q_scale or not?
        TORCH_CHECK(!options.has_dtype() || options.dtype() == self.dtype(),
                    "It is currently not supported to specify a dtype that doesn't match "
                    "the input tensor's dtype via empty_like.  Specified: ", options.dtype(),
                    " Input tensor's dtype: ", self.dtype());
        auto qscheme = self.qscheme();
        if (qscheme == kPerTensorAffine) {
          return _empty_affine_quantized(self.sizes(), options.memory_format(memory_format),
                                             self.q_scale(),
                                             self.q_zero_point(),
                                             // See Note [Explicit nullopt MemoryFormat argument]
                                             nullopt);
        } else if (qscheme == kPerChannelAffine) {
          // Copy the tensors with channels to avoid accidental overrides
          return _empty_per_channel_affine_quantized(
              self.sizes(),
              self.q_per_channel_scales().clone(MemoryFormat::Preserve),
              self.q_per_channel_zero_points().clone(MemoryFormat::Preserve),
              self.q_per_channel_axis(),
              options.memory_format(memory_format),
              // See Note [Explicit nullopt MemoryFormat argument]
              nullopt);
        } else {
          TORCH_CHECK(false, "Unsupported qscheme: ", toString(qscheme));
        }
      }

      Tensor result;

      if (memory_format == MemoryFormat::Preserve) {
        if (self.is_non_overlapping_and_dense()) {
          result = empty_strided(self.sizes(), self.strides(), options.memory_format(nullopt));
        } else if (self.unsafeGetTensorImpl()->support_as_strided() && self.layout() == kStrided) {
          // If input tensor is not dense and non-overlapping but strided, we will infer an output strides
          // which keeps the layout permutation of the input tensor.
          vector<i64> strides = infer_dense_strides(self.sizes(), self.strides());
          // See Note [Explicit nullopt MemoryFormat argument]
          result = empty_strided(self.sizes(), strides, options.memory_format(nullopt));
        } else {
          // See Note [Explicit nullopt MemoryFormat argument]
          result = empty(self.sizes(), options.memory_format(self.suggest_memory_format()), nullopt);
        }
      } else {
        // See Note [Explicit nullopt MemoryFormat argument]
        result = empty(self.sizes(), options.memory_format(memory_format), nullopt);
      }

      if (self.opt_names()) {
        namedinference::propagate_names(result, self.names());
      }

      // never propagate Conjugate key
      result._set_conj(false);
      return result;
        */
}

pub fn new_empty(
        self_:          &Tensor,
        size:           &[i32],
        dtype_opt:      Option<ScalarType>,
        layout_opt:     Option<Layout>,
        device_opt:     Option<Device>,
        pin_memory_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            auto dtype = dtype_opt.has_value() ? dtype_opt : optTypeMetaToScalarType(self.options().dtype_opt());
      auto layout = layout_opt.has_value() ? layout_opt : self.options().layout_opt();
      auto device = device_opt.has_value() ? device_opt : self.options().device_opt();
      auto pin_memory = pin_memory_opt.has_value() ? pin_memory_opt : self.options().pinned_memory_opt();
      return empty(size, dtype, layout, device, pin_memory, nullopt);
        */
}

pub fn new_empty_strided(
        self_:      &Tensor,
        size:       &[i32],
        stride:     &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      return empty_strided(size, stride, self.options().merge_in(options));
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ eye ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn eye_a(
        n:          i64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // the default value of `m` equals to `n`
      return native::eye(n, n, dtype, layout, device, pin_memory);
        */
}

pub fn eye_b(
        n:          i64,
        m:          i64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto tensor = empty({0}, options); // to be resized
      return eye_out(tensor, n, m);
        */
}


pub fn eye_out_cpu_a(
        n:      i64,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // the default value of `m` equals to `n`
      return native::eye_out_cpu(n, n, result);
        */
}

pub fn eye_out_cpu_b(
        n:      i64,
        m:      i64,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
      TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

      result.resize_({n, m});
      result.zero_();

      i64 sz = min<i64>(n, m);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, result.scalar_type(), "eye", [&]() -> void {
        Scalar* result_data = result.data_ptr<Scalar>();
        parallel_for(0, sz, internal::GRAIN_SIZE, [&](i64 p_begin, i64 p_end) {
          for(i64 i = p_begin; i < p_end; i++)
            result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
        });
      });

      return result;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ full ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | Performs dtype inference for full
  |
  */
pub fn infer_full_options(
        fill_value: &Scalar,
        options:    &TensorOptions) -> TensorOptions {
    
    todo!();
        /*
            if (!options.has_dtype()) {
        if (fill_value.isBoolean()) {
          return options.dtype(kBool);
        } else if (fill_value.isIntegral(false)) {
          return options.dtype(kLong);
        } else if (fill_value.isComplex()) {
          auto scalar_type = (get_default_dtype() == ScalarType::Double) ?
                                ScalarType::ComplexDouble :
                                ScalarType::ComplexFloat;
          return options.dtype(scalar_type);
        } else {
          return options.dtype(get_default_dtype());
        }
      }

      return options;
        */
}

pub fn full_a(
        size:       &[i32],
        fill_value: &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(options.layout() != kSparse,
        "full(...) is not implemented for sparse layout");

      auto result = empty(size, infer_full_options(fill_value, options));
      return result.fill_(fill_value);
        */
}

pub fn full_out(
        size:       &[i32],
        fill_value: &Scalar,
        result:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!result.is_sparse(),
        "full(...) is not implemented for sparse layout");

      result.resize_(size);
      return result.fill_(fill_value);
        */
}

pub fn full_like(
        self_:                  &Tensor,
        fill_value:             &Scalar,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty_like(self, options, optional_memory_format);
      return result.fill_(fill_value);
        */
}

pub fn new_full(
        self_:      &Tensor,
        size:       &[i32],
        fill_value: &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      return full(size, fill_value, self.options().merge_in(options));
        */
}

pub fn linspace_logspace_infer_options(
    start:   &Scalar,
    end:     &Scalar,
    options: &TensorOptions,
    fn_name: *const u8) -> TensorOptions {
    
    todo!();
        /*
            if (start.isComplex() || end.isComplex()) {
        const auto default_complex_dtype = get_default_complex_dtype();
        if (options.has_dtype()) {
          auto dtype = typeMetaToScalarType(options.dtype());
          TORCH_CHECK(isComplexType(dtype),
              fn_name, ": inferred dtype ", default_complex_dtype, " can't be safely cast to passed dtype ", dtype);
        } else {
          return options.dtype(default_complex_dtype);
        }
      }

      return options.has_dtype() ? options : options.dtype(get_default_dtype());
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn linspace(
        start:      &Scalar,
        end:        &Scalar,
        steps:      Option<i64>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      const auto steps_ = steps.value_or(100);
      TORCH_CHECK(steps_ >= 0, "number of steps must be non-negative");
      auto result_options = linspace_logspace_infer_options(start, end, options, "torch.linspace()");
      Tensor result = empty({steps_}, result_options);
      return linspace_out(result, start, end, steps);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ logspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn logspace(
    start:      &Scalar,
    end:        &Scalar,
    steps:      Option<i64>,
    base:       f64,
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {

    todo!();
    /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      const auto steps_ = steps.value_or(100);
      TORCH_CHECK(steps_ >= 0, "number of steps must be non-negative");
      auto result_options = linspace_logspace_infer_options(start, end, options, "torch.logspace()");
      Tensor result = empty({steps_}, result_options);
      return logspace_out(result, start, end, steps, base);
        */
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn ones_a(
    size:       &[i32],
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {

    todo!();
        /*
            return native::full(size, /*fill_value=*/1., dtype, layout, device, pin_memory);
        */
}

pub fn ones_out(
        size:   &[i32],
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::full_out(size, /*fill_value=*/1., result);
        */
}

pub fn ones_like(
        self_:                  &Tensor,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            auto result = empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
      return result.fill_(1.);
        */
}

pub fn new_ones(
        self_:      &Tensor,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options =
          TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
              pin_memory);

      return ones(size, self.options().merge_in(options));
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scalar_tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn scalar_tensor(
        s:          &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      if (options.device() == kCPU) {
        // This is a fast track to skip device dispatch for making scalar tensor on CPU.
        // See https://github.com/pytorch/pytorch/pull/29915 for more detailed perf
        // difference.
        // In the future when we remove the overhead of device dispatch, we'll happily
        // revert this to following:
        //   auto result = empty({}, options);
        tracer::NoTracerDispatchMode tracer_guard;
        AutoDispatchBelowAutograd mode;
        auto result = empty_cpu({}, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
        native::fill_(result, s);
        return result;
      }
      return empty({}, options).fill_(s);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ rand ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn rand_a(
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::rand(size, static_cast<optional<Generator>>(nullopt), dtype, layout, device, pin_memory);
        */
}

pub fn rand_b(
        size:       &[i32],
        generator:  Option<Generator>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.uniform_(0, 1, generator);
        */
}

pub fn rand_out_a(
        size:   &[i32],
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::rand_out(size, nullopt, result);
        */
}

pub fn rand_out_b(
        size:      &[i32],
        generator: Option<Generator>,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            result.resize_(size);
      return result.uniform_(0, 1, generator);
        */
}

pub fn rand_like(
        self_:                  &Tensor,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty_like(self, options, optional_memory_format);
      return result.uniform_(0, 1, nullopt);
        */
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn randint_a(
        high:       i64,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randint(high, size, nullopt /* generator*/, dtype, layout, device, pin_memory);
        */
}



pub fn randint_b(
        high:       i64,
        size:       &[i32],
        generator:  Option<Generator>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randint(0, high, size, generator, dtype, layout, device, pin_memory);
        */
}


pub fn randint_c(
        low:        i64,
        high:       i64,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randint(low, high, size, nullopt, dtype, layout, device, pin_memory);
        */
}

pub fn randint_d(
        low:        i64,
        high:       i64,
        size:       &[i32],
        generator:  Option<Generator>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.random_(low, high, generator);
        */
}



pub fn randint_out_a(
        high:   i64,
        size:   &[i32],
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::randint_out(high, size, nullopt, result);
        */
}


pub fn randint_out_b(
        high:      i64,
        size:      &[i32],
        generator: Option<Generator>,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            result.resize_(size);
      return result.random_(0, high, generator);
        */
}


pub fn randint_out_c(
        low:    i64,
        high:   i64,
        size:   &[i32],
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::randint_out(low, high, size, nullopt, result);
        */
}

pub fn randint_out_d(
        low:       i64,
        high:      i64,
        size:      &[i32],
        generator: Option<Generator>,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            result.resize_(size);
      return result.random_(low, high, generator);
        */
}


pub fn randint_like_a(
        self_:                  &Tensor,
        high:                   i64,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty_like(self, options, optional_memory_format);
      return result.random_(0, high, nullopt);
        */
}

pub fn randint_like_b(
        self_:                  &Tensor,
        low:                    i64,
        high:                   i64,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty_like(self, options, optional_memory_format);
      return result.random_(low, high, nullopt);
        */
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randn ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


pub fn randn_a(
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randn(size, static_cast<optional<Generator>>(nullopt), dtype, layout, device, pin_memory);
        */
}



pub fn randn_b(
        size:       &[i32],
        generator:  Option<Generator>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.normal_(0, 1, generator);
        */
}



pub fn randn_out_a(
        size:   &[i32],
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return native::randn_out(size, nullopt, result);
        */
}

pub fn randn_out_b(
        size:      &[i32],
        generator: Option<Generator>,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            result.resize_(size);
      return result.normal_(0, 1, generator);
        */
}

pub fn normal(
    mean:       f64,
    std:        f64,
    size:       &[i32],
    generator:  Option<Generator>,
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.normal_(mean, std, generator);
        */
}

pub fn normal_out(
    mean:      f64,
    std:       f64,
    size:      &[i32],
    generator: Option<Generator>,
    result:    &mut Tensor) -> &mut Tensor {

    todo!();
        /*
            result.resize_(size);
      return result.normal_(mean, std, generator);
        */
}

pub fn randn_like(
    self_:                  &Tensor,
    dtype:                  Option<ScalarType>,
    layout:                 Option<Layout>,
    device:                 Option<Device>,
    pin_memory:             Option<bool>,
    optional_memory_format: Option<MemoryFormat>) -> Tensor {

    todo!();
    /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty_like(self, options, optional_memory_format);
      return result.normal_(0, 1, nullopt);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randperm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn randperm_cpu<Scalar>(
    result:    &mut Tensor,
    n:         i64,
    generator: *mut CPUGeneratorImpl)  {

    todo!();
        /*
            Scalar *r__data = result.data_ptr<Scalar>();

      result.resize_({n});
      i64 r__stride_0 = result.stride(0);

      parallel_for(0, n, internal::GRAIN_SIZE,
                      [&r__data, &r__stride_0](i64 p_begin, i64 p_end) {
        for(i64 i = p_begin; i < p_end; i++)
          r__data[i*r__stride_0] = static_cast<Scalar>(i);
      });

      for(i64 i = 0; i < n - 1; i++)
      {
        i64 z = generator->random() % (n-i);
        Scalar sav = r__data[i*r__stride_0];
        r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
        r__data[(z+i)*r__stride_0] = sav;
      }
        */
}

pub fn randperm_a(
        n:          i64,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randperm(n, nullopt, dtype, layout, device, pin_memory);
        */
}

pub fn randperm_b(
        n:          i64,
        generator:  Option<Generator>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            if (!dtype.has_value()) {
        dtype = ScalarType::Long;
      }

      // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto tensor = empty(n, options);
      return randperm_out(tensor, n, generator);
        */
}

pub fn randperm_out(
        n:      i64,
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return randperm_out(result, n, nullopt);
        */
}

pub fn randperm_out_cpu(
        n:         i64,
        generator: Option<Generator>,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
      TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()), "Expected a '", result.device(), "' generator device but found '", generator->device(), "'");
      check_supported_max_int_with_precision(n, result);
      result.resize_({n});
      auto gen = get_generator_or_default<CPUGeneratorImpl>(generator, getDefaultCPUGenerator());
      // See Note [Acquire lock when using random generators]
      lock_guard<mutex> lock(gen->mutex_);
      AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, result.scalar_type(), "randperm", [&]() -> void {
        randperm_cpu<Scalar>(result, n, gen);
      });

      return result;
        */
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ range ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn range_a(
        start:      &Scalar,
        end:        &Scalar,
        step:       &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      Tensor result = empty({0}, options);
      return range_out(result, start, end, step);
        */
}

pub fn range_b(
        start:      &Scalar,
        end:        &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::range(start, end, 1, dtype, layout, device, pin_memory);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn tril_indices_cpu(
        row:            i64,
        col:            i64,
        offset:         i64,
        dtype_opt:      Option<ScalarType>,
        layout_opt:     Option<Layout>,
        device_opt:     Option<Device>,
        pin_memory_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            if (!dtype_opt.has_value()) {
        dtype_opt = ScalarType::Long;
      }

      check_args(row, col, layout_opt);

      auto tril_size = get_tril_size(row, col, offset);

      // create an empty Tensor with correct size
      auto result = native::empty_cpu({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

      // The following three approaches result in very little performance
      // differences. Hence, the 2nd option is taken for simpler code, and to return
      // contiguous tensors. Refer to #14904 for more details.
      //
      // 1. sequential RAM access: fill row coordinates first, then columns. This
      //    results in two for-loop and more arithmetic operations.
      //
      // 2. interleaved RAM access: fill in index coordinates one by one, which
      //    jumps between the two output Tensor rows in every iteration.
      //
      // 3. sequential RAM + transpose: create an n X 2 Tensor, fill the Tensor
      //    sequentially, and then transpose it.
      AT_DISPATCH_ALL_TYPES(result.scalar_type(), "tril_indices", [&]() -> void {
        // fill the Tensor with correct values
        Scalar* result_data = result.data_ptr<Scalar>();
        i64 i = 0;

        Scalar r = max<i64>(0, -offset), c = 0;
        while (i < tril_size) {
          result_data[i] = r;
          result_data[tril_size + i++] = c;

          // move to the next column and check if (r, c) is still in bound
          c += 1;
          if (c > r + offset || c >= col) {
            r += 1;
            c = 0;
            // NOTE: not necessary to check if r is less than row here, because i
            // and tril_size provide the guarantee
          }
        }
      });

      return result;
        */
}

pub fn triu_indices_cpu(
        row:            i64,
        col:            i64,
        offset:         i64,
        dtype_opt:      Option<ScalarType>,
        layout_opt:     Option<Layout>,
        device_opt:     Option<Device>,
        pin_memory_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            if (!dtype_opt.has_value()) {
        dtype_opt = ScalarType::Long;
      }

      check_args(row, col, layout_opt);

      auto triu_size = row * col - get_tril_size(row, col, offset - 1);

      // create an empty Tensor with correct size
      auto result = native::empty_cpu({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

      AT_DISPATCH_ALL_TYPES(result.scalar_type(), "triu_indices", [&]() -> void {
        // fill the Tensor with correct values
        Scalar* result_data = result.data_ptr<Scalar>();
        i64 i = 0;
        // not typing max with Scalar as it could be an unsigned type
        // NOTE: no need to check if the returned value of max overflows
        // Scalar, as i and triu_size act as a guard.
        Scalar c = max<i64>(0, offset), r = 0;
        while (i < triu_size) {
          result_data[i] = r;
          result_data[triu_size + i++] = c;

          // move to the next column and check if (r, c) is still in bound
          c += 1;
          if (c >= col) {
            r += 1;
            // not typing max with Scalar as it could be an unsigned type
            // NOTE: not necessary to check if c is less than col or overflows here,
            // because i and triu_size act as a guard.
            c = max<i64>(0, r + offset);
          }
        }
      });

      return result;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ zeros ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn zeros_a(
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.zero_();
        */
}

pub fn zeros_out(
        size:   &[i32],
        result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (result.is_sparse()) {
        result.sparse_resize_and_clear_(size, size.size(), 0.);
        return result;
      } else {
        result.resize_(size);
      }
      return result.zero_();
        */
}

pub fn zeros_like(
        self_:                  &Tensor,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      if (options.layout() == kSparse && self.is_sparse()) {
        TORCH_CHECK(
            !(optional_memory_format.has_value()),
            "memory format option is only supported by strided tensors");
        auto res = empty({0}, options); // to be resized
        res.sparse_resize_and_clear_(
            self.sizes(), self.sparse_dim(), self.dense_dim());
        return res;
      }
      auto result = empty_like(self, options, optional_memory_format);
      return result.zero_();
        */
}

pub fn new_zeros(
        self_:      &Tensor,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      return zeros(size, self.options().merge_in(options));
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ bartlett_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


pub fn bartlett_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::bartlett_window(
          window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn bartlett_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("bartlett_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return native::ones({1}, dtype, layout, device, pin_memory);
      }
      if (periodic) {
        window_length += 1;
      }
      auto window = native::arange(window_length, dtype, layout, device, pin_memory)
                        .mul_(2. / static_cast<double>(window_length - 1));
      const i64 first_half_size = ((window_length - 1) >> 1) + 1;
      window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ blackman_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn blackman_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::blackman_window(
          window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn blackman_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("blackman_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return native::ones({1}, dtype, layout, device, pin_memory);
      }
      if (periodic) {
        window_length += 1;
      }
      // from https://en.wikipedia.org/wiki/Window_function#Blackman_window
      auto window =
          native::arange(window_length, dtype, layout, device, pin_memory)
              .mul_(pi<double> / static_cast<double>(window_length - 1));
      window = window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn hamming_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hamming_window(
          window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn hamming_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hamming_window(
          window_length,
          periodic,
          /*alpha=*/0.54,
          dtype,
          layout,
          device,
          pin_memory);
        */
}

pub fn hamming_window_c(
        window_length: i64,
        periodic:      bool,
        alpha:         f64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hamming_window(
          window_length, periodic, alpha, /*beta=*/0.46, dtype, layout, device, pin_memory);
        */
}

pub fn hamming_window_d(
        window_length: i64,
        periodic:      bool,
        alpha:         f64,
        beta:          f64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("hamming_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return native::ones({1}, dtype, layout, device, pin_memory);
      }
      if (periodic) {
        window_length += 1;
      }
      auto window = native::arange(window_length, dtype, layout, device, pin_memory);
      window.mul_(pi<double> * 2. / static_cast<double>(window_length - 1)).cos_().mul_(-beta).add_(alpha);
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn hann_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hann_window(window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn hann_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("hann_window", options, window_length);
      return native::hamming_window(
          window_length, periodic, /*alpha=*/0.5, /*beta=*/0.5, dtype, layout, device, pin_memory);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ kaiser_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn kaiser_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::kaiser_window(
          window_length,
          /*periodic=*/true,
          /*beta=*/12.0,
          dtype,
          layout,
          device,
          pin_memory);
        */
}

pub fn kaiser_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::kaiser_window(window_length, periodic, /*beta=*/12.0, dtype, layout, device, pin_memory);
        */
}

pub fn kaiser_window_c(
        window_length: i64,
        periodic:      bool,
        beta:          f64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("kaiser_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return ones({1}, options);
      }
      if (periodic) {
        window_length += 1;
      }
      auto initial = arange(window_length, options);
      auto window = empty(window_length, options);
      auto iter = TensorIterator::unary_op(window, initial);
      kaiser_window_stub(iter.device_type(), iter, window_length, beta);
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ vandermonde_matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn vander(
        x:          &Tensor,
        N:          Option<i64>,
        increasing: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(x.dim() == 1, "x must be a one-dimensional tensor.");

      // Acquires n, defaulting to size if not provided
      i64 n = x.size(0);
      if (N.has_value()) {
        n = *N;
        TORCH_CHECK(n >= 0, "N must be non-negative.");
      }

      // Note: result is long if x is an integer tensor (like int8) because
      // cumprod promotes integer tensors to long
      auto result = empty({x.size(0), n}, x.options().dtype(promote_types(x.scalar_type(), ScalarType::Long)));

      if (n > 0) {
        result.select(1, 0).fill_(1);
      }
      if (n > 1) {
        result.slice(1, 1).copy_(x.unsqueeze(1));
        result.slice(1, 1).copy_(cumprod(result.slice(1, 1), 1));
      }

      if (!increasing) {
        return flip(result, {1});
      }
      return result;
        */
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn tensor_cpu<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_cpu(values, options);
        */
}


pub fn tensor_backend<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_backend(values, options);
        */
}

pub fn tensor_complex_cpu<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_complex_cpu(values, options);
        */
}


pub fn tensor_complex_backend<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_complex_backend(values, options);
        */
}

pub fn from_file(
        filename:   StringView,
        shared:     Option<bool>,
        size:       Option<i64>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

        TORCH_CHECK(!options.pinned_memory(), "tensors constructed from a file cannot be pinned");
        i64 my_size = size.value_or(0);
        int flags = shared.value_or(false) ? TH_ALLOCATOR_MAPPED_SHARED : 0;
        auto my_dtype = options.dtype();
        usize size_bytes = my_size * my_dtype.itemsize();
        auto storage_impl = make_intrusive<StorageImpl>(
            StorageImpl::use_byte_Size(),
            size_bytes,
            THMapAllocator::makeDataPtr(
                string(filename), flags, size_bytes, nullptr),
            /*allocator=*/nullptr,
            /*resizable=*/false);
        auto tensor = make_tensor<TensorImpl>(
            storage_impl, DispatchKey::CPU, my_dtype);
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous({my_size});
        return tensor;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ clone ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn clone(
        src:                    &Tensor,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            auto memory_format =
          optional_memory_format.value_or(MemoryFormat::Preserve);
      if (memory_format == MemoryFormat::Preserve) {
        if (src.is_non_overlapping_and_dense()) {
          // Copy all strides
          auto self = empty_strided(src.sizes(), src.strides(), src.options());
          self.copy_(src);
          return self;
        } else {
          memory_format = src.suggest_memory_format();
        }
      }
      auto self = empty_like(src, src.options(), memory_format);
      self.copy_(src);
      return self;
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~ named tensor overloads ~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | In the short term, these exist.
  |
  | In the long term, we should move DimnameList
  | into TensorOptions to avoid having these
  | overloads.
  |
  */
pub fn full_b(
        size:       &[i32],
        fill_value: &Scalar,
        names:      Option<DimnameList>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(options.layout() != kSparse,
        "full(...) is not implemented for sparse layout");

      auto result = empty(size, names, infer_full_options(fill_value, options));
      return result.fill_(fill_value);
        */
}

pub fn ones_b(
        size:       &[i32],
        names:      Option<DimnameList>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]

      return native::full(
          size, /*fill_value=*/1., names, dtype, layout, device, pin_memory);
        */
}

pub fn zeros_b(
        size:       &[i32],
        names:      Option<DimnameList>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::full(size, /*fill_value=*/0., names, dtype, layout, device, pin_memory);
        */
}


pub fn randn_c(
        size:       &[i32],
        names:      Option<DimnameList>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randn(size, nullopt, names, dtype, layout, device, pin_memory);
        */
}

pub fn randn_d(
        size:       &[i32],
        generator:  Option<Generator>,
        names:      Option<DimnameList>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, names, options);
      return result.normal_(0, 1, generator);
        */
}


pub fn rand_c(
        size:       &[i32],
        names:      Option<DimnameList>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::rand(size, nullopt, names, dtype, layout, device, pin_memory);
        */
}

pub fn rand_d(
        size:       &[i32],
        generator:  Option<Generator>,
        names:      Option<DimnameList>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, names, options);
      return result.uniform_(0, 1, generator);
        */
}

define_dispatch!{kaiser_window_stub}
