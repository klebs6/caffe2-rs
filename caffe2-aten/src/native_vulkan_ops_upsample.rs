// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Upsample.cpp]

pub fn upsample_nearest2d(
    input_arg:    &Tensor,
    output_sizes: &[i32],
    scales_h:     Option<f64>,
    scales_w:     Option<f64>) -> Tensor {
    
    todo!();
        /*
            api::Context* const context = api::context();

      const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
      const vTensor& v_input = convert(input);
      const auto v_input_sizes = v_input.sizes();

      TORCH_CHECK(
          (4 == v_input_sizes.size()) && (2 == output_sizes.size()),
          "Invalid input!");

      vTensor v_output{
        context,
        {
          v_input_sizes[Layout::Activation4D::batch],
          v_input_sizes[Layout::Activation4D::channels],
          output_sizes[Layout::Parameter::height],
          output_sizes[Layout::Parameter::width],
        },
        input.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_input.has_image()) {
          const struct Block final {
            uvec3 extents;
            u32 _;
            ivec2 iextents;
            vec2 scale;
          } block {
            v_output.extents(),
            0u,
            {
              safe_downcast<i32>(input.size(Layout::Activation4D::width) - 1),
              safe_downcast<i32>(input.size(Layout::Activation4D::height) - 1),
            },
            {
                compute_scales_value<float>(
                    scales_w,
                    v_input_sizes[Layout::Activation4D::width],
                    output_sizes[Layout::Parameter::width]),
                compute_scales_value<float>(
                    scales_h,
                    v_input_sizes[Layout::Activation4D::height],
                    output_sizes[Layout::Parameter::height]),
            },
          };

          context->dispatch(
              command_buffer,
              {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              VK_KERNEL(upsample_nearest2d),
              v_output.extents(),
              context->gpu().adapter->local_work_group_size(),
              // Write-only access bypasses synchronization but inserts appropriate
              // barriers if necessary.
              v_output.image(
                  command_buffer,
                  vTensor::Stage::Compute,
                  vTensor::Access::Write),
              // Read-only access is implied on const tensors and triggers an async
              // synchronization if necessary.
              v_input.image(
                  command_buffer,
                  vTensor::Stage::Compute),
              // Object lifetime is managed by the resource pool.
              // It is OK not to keep track of the handle.
              context->resource().pool.uniform(block).object);
        }
        else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return convert(v_output);
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d));
    }
    */
}
