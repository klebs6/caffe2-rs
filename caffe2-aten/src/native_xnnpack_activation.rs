// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/Activation.cpp]

pub fn use_hardswish(input: &Tensor) -> bool {
    
    todo!();
        /*
            return xnnpack::internal::available() &&
              (1 <= input.ndimension()) &&
              (input.device().is_cpu()) &&
              (kFloat == input.scalar_type()) &&
              !input.requires_grad() &&
               true;
        */
}

pub fn hardswish_impl(
        input:  &mut Tensor,
        output: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            using namespace internal;

      xnn_operator_t hardswish_op{};
      const xnn_status create_status = xnn_create_hardswish_nc_f32(
        1, // channels
        1, // input stride
        1, // output stride
        0, // flags
        &hardswish_op);

      TORCH_CHECK(
        xnn_status_success == create_status,
        "xnn_create_hardswish_nc_f32 failed!");

      Operator hardswish_scoped_op(hardswish_op);

      const xnn_status setup_status = xnn_setup_hardswish_nc_f32(
        hardswish_op,
        input.numel(),  // Batch
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        pthreadpool_());  // threadpool

      TORCH_CHECK(
        xnn_status_success == setup_status,
        "xnn_setup_hardswish_nc_f32 failed!");

      const xnn_status run_status = xnn_run_operator(
        hardswish_op,
        pthreadpool_());  // threadpool

      TORCH_INTERNAL_ASSERT(
        xnn_status_success == run_status,
        "xnn_run_operator failed!");

      return output;
        */
}

pub fn hardswish(input: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
        input, input.suggest_memory_format());

      Tensor output = mobile::empty_with_tail_padding(
        padded_input.sizes(),
        padded_input.options().dtype(),
        input.suggest_memory_format(),
        padded_input.names());

      hardswish_impl(padded_input, output);
      return output.contiguous(input.suggest_memory_format());
        */
}

pub fn hardswish_mut(input: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            using namespace internal;

      hardswish_impl(input, input);
      return input;
        */
}
