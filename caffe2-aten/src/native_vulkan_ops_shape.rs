crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Shape.cpp]

pub fn view(
        self_arg: &Tensor,
        shape:    &[i32]) -> Tensor {
    
    todo!();
        /*
            api::Context* const context = api::context();

      const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
      const vTensor& v_self = convert(self);

      vTensor v_output{
        context,
        shape,
        self.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        command_buffer.copy(
            // Read-only access is implied on const tensors and triggers an async
            // synchronization if necessary.
            v_self.buffer(
                command_buffer,
                vTensor::Stage::Transfer),
            // Write-only access bypasses synchronization but inserts appropriate
            // barriers if necessary.
            v_output.buffer(
                command_buffer,
                vTensor::Stage::Transfer,
                vTensor::Access::Write));
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return convert(v_output);
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("view", TORCH_FN(view));
    }
    */
}
