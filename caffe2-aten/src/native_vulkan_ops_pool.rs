// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Pool.cpp]

pub fn adaptive_avg_pool2d(
    self_arg:    &Tensor,
    output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          self_arg.dim() == 4,
          "Vulkan adaptive_avg_pool2d expects 4-dimensional input!");

      api::Context* const context = api::context();

      const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
      const vTensor& v_self = convert(self);

      vTensor v_output{
        context,
        {
          self.size(Layout::Activation4D::batch),
          self.size(Layout::Activation4D::channels),
          output_size[Layout::Activation4D::batch],
          output_size[Layout::Activation4D::channels],
        },
        v_self.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_self.has_image()) {
          const uvec3 v_output_size = v_output.extents();
          const uvec3 v_self_size = v_self.extents();

          const vec2 stride {
            static_cast<float>(v_self_size.data[0u]) / v_output_size.data[0u],
            static_cast<float>(v_self_size.data[1u]) / v_output_size.data[1u],
          };

          const struct Block final {
            uvec3 extents;
            u32 _;
            vec2 kernel;
            vec2 stride;
          } block {
            v_output.extents(),
            0u,
            {
              v_self_size.data[0u] - (v_output_size.data[0u] - 1u) * stride.data[0u],
              v_self_size.data[1u] - (v_output_size.data[1u] - 1u) * stride.data[1u],
            },
            stride,
          };

          context->dispatch(
              command_buffer,
              {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              VK_KERNEL(adaptive_avg_pool2d),
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
              v_self.image(
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

pub fn pool2d(
        self_arg:          &Tensor,
        kernel_arg:        &[i32],
        stride_arg:        &[i32],
        padding_arg:       &[i32],
        dilation_arg:      &[i32],
        ceil_mode:         bool,
        shader_descriptor: &ShaderDescriptor) -> Tensor {
    
    todo!();
        /*
            if (stride_arg.empty()) {
        stride_arg = kernel_arg;
      }

      TORCH_CHECK(!kernel_arg.empty(), "Kernel size cannot be empty!");
      TORCH_CHECK(!stride_arg.empty(), "Stride cannot be empty!");
      TORCH_CHECK(!padding_arg.empty(), "Padding cannot be empty!");

      static const auto normalize = [](const IntArrayRef parameter) {
        return array<i64, 2>{
          parameter[0],
          (2 == parameter.size()) ? parameter[1] : parameter[0],
        };
      };

      const auto input_size = self_arg.sizes();
      const auto kernel = normalize(kernel_arg);
      const auto stride = normalize(stride_arg);
      const auto padding = normalize(padding_arg);
      const auto dilation = normalize(dilation_arg);

      const i64 output_height = pooling_output_shape(
          input_size[Layout::Activation4D::height],
          kernel[Layout::Parameter::height],
          padding[Layout::Parameter::height],
          stride[Layout::Parameter::height],
          dilation[Layout::Parameter::height],
          ceil_mode);

      const i64 output_width = pooling_output_shape(
          input_size[Layout::Activation4D::width],
          kernel[Layout::Parameter::width],
          padding[Layout::Parameter::width],
          stride[Layout::Parameter::width],
          dilation[Layout::Parameter::width],
          ceil_mode);

      pool2d_shape_check(
          self_arg,
          kernel[Layout::Parameter::height],
          kernel[Layout::Parameter::width],
          stride[Layout::Parameter::height],
          stride[Layout::Parameter::width],
          padding[Layout::Parameter::height],
          padding[Layout::Parameter::width],
          dilation[Layout::Parameter::height],
          dilation[Layout::Parameter::width],
          input_size[Layout::Activation4D::channels],
          input_size[Layout::Activation4D::height],
          input_size[Layout::Activation4D::width],
          output_height,
          output_width,
          self_arg.suggest_memory_format());

      api::Context* const context = api::context();

      const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
      const vTensor& v_self = convert(self);

      vTensor v_output{
        context,
        {
          input_size[Layout::Activation4D::batch],
          input_size[Layout::Activation4D::channels],
          output_height,
          output_width,
        },
        v_self.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_self.has_image()) {
          const struct Block final {
            uvec3 extents;
            i32 range;
            ivec4 kernel;
            ivec2 stride;
            ivec2 padding;
            ivec2 dilation;
          } block {
            v_output.extents(),
            safe_downcast<i32>(
                kernel[Layout::Parameter::width] *
                kernel[Layout::Parameter::height]),
            {
              safe_downcast<i32>(kernel[Layout::Parameter::width]),
              safe_downcast<i32>(kernel[Layout::Parameter::height]),
              safe_downcast<i32>(self.size(Layout::Activation4D::width)),
              safe_downcast<i32>(self.size(Layout::Activation4D::height)),
            },
            {
              safe_downcast<i32>(stride[Layout::Parameter::width]),
              safe_downcast<i32>(stride[Layout::Parameter::height]),
            },
            {
              safe_downcast<i32>(padding[Layout::Parameter::width]),
              safe_downcast<i32>(padding[Layout::Parameter::height]),
            },
            {
              safe_downcast<i32>(dilation[Layout::Parameter::width]),
              safe_downcast<i32>(dilation[Layout::Parameter::height]),
            },
          };

          context->dispatch(
              command_buffer,
              {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              shader_descriptor,
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
              v_self.image(
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


pub fn avg_pool2d(
        self_arg:          &Tensor,
        kernel_arg:        &[i32],
        stride_arg:        &[i32],
        padding_arg:       &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {
    
    todo!();
        /*
            return pool2d(
        self_arg,
        kernel_arg,
        stride_arg,
        padding_arg,
        {1,1},
        ceil_mode,
        VK_KERNEL(avg_pool2d)
      );
        */
}

pub fn max_pool2d(
        self_arg:     &Tensor,
        kernel_arg:   &[i32],
        stride_arg:   &[i32],
        padding_arg:  &[i32],
        dilation_arg: &[i32],
        ceil_mode:    bool) -> Tensor {
    
    todo!();
        /*
            return pool2d(
        self_arg,
        kernel_arg,
        stride_arg,
        padding_arg,
        dilation_arg,
        ceil_mode,
        VK_KERNEL(max_pool2d)
      );
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("_adaptive_avg_pool2d", TORCH_FN(adaptive_avg_pool2d));
      m.impl("avg_pool2d", TORCH_FN(avg_pool2d));
      m.impl("max_pool2d", TORCH_FN(max_pool2d));
    }
    */
}
