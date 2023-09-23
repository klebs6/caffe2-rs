// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorConversions.cpp]

/**
  | Take a Device that may not have device_index
  | set (i.e., having it as -1 representing the
  | current device) and return the corresponding
  | Device according to the actual device at the
  | time of this function call.
  |
  | No-op if the device_index is set.
  |
  */
#[inline] pub fn ensure_has_index(device: Device) -> Device {
    
    todo!();
        /*
            if (device.is_cpu() || device.has_index()) {
        return device;
      }
      const DeviceGuardImplInterface* impl = getDeviceGuardImpl(device.type());
      return impl->getDevice();
        */
}

#[inline] pub fn to_impl(
        self_:        &Tensor,
        options:      &TensorOptions,
        non_blocking: bool,
        copy_:        bool) -> Tensor {
    
    todo!();
        /*
            auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);

      if (self.dtype() == options.dtype() && self.layout() == options.layout() &&
          self.device() == options.device() && !copy &&
          (memory_format == MemoryFormat::Preserve ||
           self.suggest_memory_format() == memory_format)) {
        return self;
      }

      bool pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() &&
                      (options.layout() == kStrided));

      if (memory_format == MemoryFormat::Preserve) {
        if (self.is_non_overlapping_and_dense() && options.device().supports_as_strided()) {
          // Copy all strides
          auto r = empty_strided(self.sizes(),
                                     self.strides(),
                                     options.memory_format(nullopt).pinned_memory(pin_out));
          r.copy_(self, non_blocking);
          return r;
        } else {
          memory_format = self.suggest_memory_format();
        }
      }
      // See Note [Explicit nullopt MemoryFormat argument]
      auto r = empty(self.sizes(),
                         options.memory_format(memory_format).pinned_memory(pin_out),
                         nullopt);
      r.copy_(self, non_blocking);
      return r;
        */
}

pub fn to_a(
    self_:                  &Tensor,
    dtype:                  Option<ScalarType>,
    layout:                 Option<Layout>,
    device:                 Option<Device>,
    pin_memory:             Option<bool>,
    non_blocking:           bool,
    copy_:                  bool,
    optional_memory_format: Option<MemoryFormat>) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(
        !(options_.has_memory_format() && optional_memory_format.has_value()),
        "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
        "the redundant setter.");
      auto options = options_.merge_memory_format(optional_memory_format);

      TORCH_CHECK(options.requires_grad_opt() == nullopt,
               "to(options) expects unset requires_grad flag, but got "
               "options.requires_grad set as ", options.requires_grad());

      TORCH_CHECK(!options.has_layout() || self.layout() == options.layout(),
               "to(options) doesn't support converting to a different layout, "
               "but got self.layout being ", self.layout(),
               " and options.layout set as ", options.layout());

      if (options.has_device()) {
        options = options.device(ensure_has_index(options.device()));
      }
      auto specified_options = self.options().merge_in(options);
      return to_impl(self, specified_options, non_blocking, copy);
        */
}

pub fn to_b(
        self_:                  &Tensor,
        device:                 Device,
        dtype:                  ScalarType,
        non_blocking:           bool,
        copy_:                  bool,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            device = ensure_has_index(device);
      return to_impl(
          self,
          self.options().device(device).dtype(dtype).memory_format(optional_memory_format),
          non_blocking,
          copy);
        */
}

pub fn to_c(
        self_:                  &Tensor,
        dtype:                  ScalarType,
        non_blocking:           bool,
        copy_:                  bool,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            return to_impl(
          self, self.options().dtype(dtype).memory_format(optional_memory_format), non_blocking, copy);
        */
}


pub fn to_d(
        self_:                  &Tensor,
        other:                  &Tensor,
        non_blocking:           bool,
        copy_:                  bool,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            auto options = other.options();
      return to_impl(self, options.memory_format(optional_memory_format), non_blocking, copy);
        */
}

/**
  | This op is important primarily for lazy
  | / graph-based backends.
  |
  | While this vanilla implementation loops through
  | each tensor and independently converts it to
  | cpu, a lazy backend like XLA might need to tell
  | sync updates across tensors.
  |
  */
pub fn to_cpu(tensors: &[Tensor]) -> Vec<Tensor> {
    
    todo!();
        /*
            vector<Tensor> cpu_tensors;
        for (const auto& t : tensors) {
            cpu_tensors.push_back(t.cpu());
        }
        return cpu_tensors;
        */
}

pub fn to_dense_backward(
        grad:  &Tensor,
        input: &Tensor) -> Tensor {
    
    todo!();
        /*
            AT_ASSERT(input_.layout() != kStrided);
      if (input_.layout() == kSparse) {
        auto input = input_.coalesce();
        return grad.sparse_mask(input);
      } else if (input_.layout() == kMkldnn) {
        return grad.to_mkldnn(input_.scalar_type());
      } else {
        AT_ERROR("Unsupported input layout: ", input_.layout());
      }
        */
}

pub fn to_mkldnn_backward(
        grad:  &Tensor,
        input: &Tensor) -> Tensor {
    
    todo!();
        /*
            AT_ASSERT(input_.layout() == kStrided);
      return grad.to_dense(input_.scalar_type());
        */
}

pub fn view_dtype(
        self_: &Tensor,
        dtype: ScalarType) -> Tensor {
    
    todo!();
        /*
            if (self.scalar_type() == dtype) {
        return self;
      }
      auto type_meta = scalarTypeToTypeMeta(dtype);
      TORCH_CHECK(self.element_size() == type_meta.itemsize(),
        "Viewing a tensor as a new dtype with a different number of bytes per element is not supported.");
      Storage storage = self.storage();
      auto new_tensor = make_tensor<TensorImpl>(
          move(storage), self.key_set(), type_meta);
      auto* impl = new_tensor.unsafeGetTensorImpl();
      impl->set_storage_offset(self.storage_offset());
      impl->set_sizes_and_strides(self.sizes(), self.strides());
      return new_tensor;
        */
}
