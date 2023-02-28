// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Mean.cpp]

pub fn mean(
    input_arg: &Tensor,
    dim:       &[i32],
    keepdim:   bool,
    dtype:     Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          input_arg.dim() == 4,
          "Vulkan mean expects 4-dimensional input!");

      static const unordered_set<i64> expected_dims_set({2, 3});
      unordered_set<i64> dims_set;

      for (const auto& d : dim) {
        dims_set.insert(utils::normalize(d, 4));
      }

      TORCH_CHECK(
          dims_set == expected_dims_set,
          "Vulkan mean currently only supports image-wide reduction!");

      api::Context* const context = api::context();

      const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
      const vTensor& v_input = convert(input);
      const IntArrayRef v_input_sizes = v_input.sizes();

      SmallVector<i64, 4u> output_sizes{
        v_input_sizes[Layout::Activation4D::batch],
        v_input_sizes[Layout::Activation4D::channels],
      };

      if (keepdim) {
        output_sizes.push_back(1);
        output_sizes.push_back(1);
      }

      vTensor v_output{
        context,
        output_sizes,
        v_input.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_input.has_image()) {
          const struct Block final {
            uvec3 extents;
            i32 range;
            uvec3 iextents;
          } block {
            v_output.extents(),
            safe_downcast<i32>(
                v_input_sizes[Layout::Activation4D::width] *
                v_input_sizes[Layout::Activation4D::height]),
            v_input.extents()
          };

          context->dispatch(
              command_buffer,
              {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              keepdim ? VK_KERNEL(mean) : VK_KERNEL(mean2d),
              v_input.extents(),
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
      m.impl("mean.dim", TORCH_FN(mean));
    }
    */
}
