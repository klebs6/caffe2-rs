/*!
 | [Note: hacky wrapper removal for optional
 | tensor]
 |
 | The kernel implementation takes an optional
 | tensor marked in the schema as Tensor? but the
 | C++ function takes Tensor instead of the
 | optional<Tensor> expected by the dispatcher.
 |
 | To remove the hacky wrapper, the C++ function
 | is changed to take optional<Tensor> and unwrap
 | the Tensor value at the beginning of the
 | function, e.g.:
 |
 |   > MaybeOwned<Tensor> weight_maybe_owned =
 |   >     borrow_from_optional_tensor(weight_opt);
 |   > const Tensor& weight = *weight_maybe_owned;
 |
 | We may want to make the kernel handle optional
 | directly without going through the creation of
 | a default-constructed Tensor in
 | borrow_from_optional_tensor.
 |
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/adaption.h]

/*
 | [Note: hacky wrapper removal for TensorOptions]
 |
 | The kernel implementation takes a TensorOptions
 | argument but the dispatcher expects separate
 | arguments for dtype, layout, device,
 | pin_memory.
 |
 | To remove the hacky wrapper, the kernel
 | implementation is changed to take the
 | 4 arguments (dtype, layout, device,
 | pin_memory), and assemble the TensorOptions
 | value at the beginning of the function, e.g.:
 |
 |   > TensorOptions options = TensorOptions().dtype(dtype).layout(layout)
 |   >    .device(device).pinned_memory(pin_memory);
 |
 | We may want make the kernel handle these
 | parameters directly without going through the
 | creation of a TensorOptions value.
 */
#[inline] pub fn check_tensor_options_and_extract_memory_format(
    options:       &TensorOptions,
    memory_format: Option<MemoryFormat>) -> Option<MemoryFormat> {

    todo!();
        /*
            TORCH_CHECK(
          options.requires_grad_opt() == nullopt ||
              options.requires_grad_opt().value() == false,
          "Operators taking TensorOptions cannot take a TensorOptions with "
          "options.requires_grad set as true. This isn't implemented yet.");
      TORCH_CHECK(
          !(options.has_memory_format() && memory_format.has_value()),
          "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
          "the redundant setter.");
      if (memory_format.has_value()) {
        return memory_format;
      } else {
        return options.memory_format_opt();
      }
        */
}

pub fn common_device_check_failure(
        common_device: &mut Option<Device>,
        tensor:        &Tensor,
        method_name:   CheckedFrom,
        arg_name:      CheckedFrom)  {
    
    todo!();
        /*
        
        */
}

#[inline] pub fn check_and_update_common_device_a(
    common_device: &mut Option<Device>,
    tensor:        &Tensor,
    method_name:   CheckedFrom,
    arg_name:      CheckedFrom)  {
    
    todo!();
        /*
            // TODO: Remove this once the following issue is addressed:
      // https://github.com/pytorch/pytorch/issues/57380
      if (!tensor.defined()) {
        return;
      }

      if (!common_device.has_value()) {
        common_device = tensor.device();
        return;
      }

      if (C10_UNLIKELY(common_device != tensor.device())) {
        common_device_check_failure(common_device, tensor, methodName, argName);
      }
        */
}

#[inline] pub fn check_and_update_common_device_b(
    common_device: &mut Option<Device>,
    tensor:        &Option<Tensor>,
    method_name:   CheckedFrom,
    arg_name:      CheckedFrom)  {

    todo!();
        /*
            if (tensor.has_value()) {
        check_and_update_common_device(common_device, tensor.value(), methodName, argName);
      }
        */
}

#[inline] pub fn check_and_update_common_device_c(
    common_device: &mut Option<Device>,
    tensors:       &[Tensor],
    method_name:   CheckedFrom,
    arg_name:      CheckedFrom)  {
    
    todo!();
        /*
            for (const auto& tensor : tensors) {
        check_and_update_common_device(common_device, tensor, methodName, argName);
      }
        */
}

#[inline] pub fn check_and_update_common_device_d(
    common_device: &mut Option<Device>,
    tensors:       &List<Option<Tensor>>,
    method_name:   CheckedFrom,
    arg_name:      CheckedFrom)  {
    
    todo!();
        /*
            for (const auto& tensor : tensors) {
        check_and_update_common_device(common_device, tensor, methodName, argName);
      }
        */
}
