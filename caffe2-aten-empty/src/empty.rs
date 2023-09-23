crate::ix!();

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
        names:                  Option<&[Dimname]>,
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

pub fn empty_out<'a>(
        size:                   &[i32],
        optional_memory_format: Option<MemoryFormat>,
        result:                 &mut Tensor) -> &'a mut Tensor {
    
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
