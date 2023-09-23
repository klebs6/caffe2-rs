// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Clamp.cpp]

pub fn clamp(
    self_arg: &Tensor,
    min:      &Option<Scalar>,
    max:      &Option<Scalar>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          min || max,
          "At least one of 'min' or 'max' must not be None");

      api::Context* const context = api::context();

      const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
      const vTensor& v_self = convert(self);

      vTensor v_output{
        context,
        v_self.sizes(),
        v_self.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
          const struct Block final {
            uvec3 extents;
            u32 _;
            vec2 clamp;
          } block {
            v_output.extents(),
            0u,
            {
              min ? min->to<float>() : -numeric_limits<float>::infinity(),
              max ? max->to<float>() : numeric_limits<float>::infinity(),
            },
          };

          context->dispatch(
              command_buffer,
              {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              VK_KERNEL(clamp),
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

pub fn clamp_mut<'a>(
    self_: &mut Tensor,
    min:   &Option<Scalar>,
    max:   &Option<Scalar>) -> &'a mut Tensor {
    
    todo!();
        /*
            api::Context* const context = api::context();

      TORCH_CHECK(
          min || max,
          "At least one of 'min' or 'max' must not be None");

      TORCH_CHECK(
          self.is_vulkan(),
          "Vulkan: In-place clamp is only supported on Vulkan tensors.");

      vTensor& v_self = convert(self);

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_self.has_image()) {
          const struct Block final {
            uvec3 extents;
            u32 _;
            vec2 clamp;
          } block {
            v_self.extents(),
            0u,
            {
              min ? min->to<float>() : -numeric_limits<float>::infinity(),
              max ? max->to<float>() : numeric_limits<float>::infinity(),
            },
          };

          context->dispatch(
              command_buffer,
              {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              VK_KERNEL(clamp_),
              v_self.extents(),
              context->gpu().adapter->local_work_group_size(),
              // Read-Write access triggers an async synchronization if necessory
              // and inserts appropriate barriers if hazards are detected.
              v_self.image(
                  command_buffer,
                  vTensor::Stage::Compute,
                  vTensor::Access::Read | vTensor::Access::Write),
              // Object lifetime is managed by the resource pool.
              // It is OK not to keep track of the handle.
              context->resource().pool.uniform(block).object);
        }
        else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return self;
        */
}

pub fn activation(
    self_arg:          &Tensor,
    shader_descriptor: &ShaderDescriptor) -> Tensor {
    
    todo!();
        /*
            api::Context* const context = api::context();

      const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
      const vTensor& v_self = convert(self);

      vTensor v_output{
        context,
        v_self.sizes(),
        v_self.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
          const struct Block final {
            uvec3 extents;
            u32 _;
          } block {
            v_output.extents(),
            0u,
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

pub fn activation_mut<'a>(
    self_:             &mut Tensor,
    shader_descriptor: &ShaderDescriptor) -> &'a mut Tensor {

    todo!();
        /*
            api::Context* const context = api::context();

      TORCH_CHECK(
          self.is_vulkan(),
          "Vulkan: In-place clamp is only supported on Vulkan tensors.");

      vTensor& v_self = convert(self);

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY(v_self.has_image()) {
          const struct Block final {
            uvec3 extents;
            u32 _;
          } block {
            v_self.extents(),
            0u,
          };

          context->dispatch(
              command_buffer,
              {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              shader_descriptor,
              v_self.extents(),
              context->gpu().adapter->local_work_group_size(),
              // Read-Write access triggers an async synchronization if necessory
              // and inserts appropriate barriers if hazards are detected.
              v_self.image(
                  command_buffer,
                  vTensor::Stage::Compute,
                  vTensor::Access::Read | vTensor::Access::Write),
              // Object lifetime is managed by the resource pool.
              // It is OK not to keep track of the handle.
              context->resource().pool.uniform(block).object);
        }
        else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return self;
        */
}

pub fn hardtanh(
    self_: &Tensor,
    min:   &Scalar,
    max:   &Scalar) -> Tensor {
    
    todo!();
        /*
            return ops::clamp(self, min, max);
        */
}

pub fn hardtanh_mut<'a>(
    self_: &mut Tensor,
    min:   &Scalar,
    max:   &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return ops::clamp_(self, min, max);
        */
}

pub fn relu<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return ops::clamp(self, 0, nullopt);
        */
}

pub fn relu_mut(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return ops::clamp_(self, 0, nullopt);
        */
}

pub fn hardswish<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return ops::activation(self, VK_KERNEL(hardswish));
        */
}

pub fn hardswish_mut(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return ops::activation_(self, VK_KERNEL(hardswish_));
        */
}

pub fn hardsigmoid<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return ops::activation(self, VK_KERNEL(hardsigmoid));
        */
}

pub fn hardsigmoid_mut(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return ops::activation_(self, VK_KERNEL(hardsigmoid_));
        */
}

pub fn sigmoid<'a>(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return ops::activation(self, VK_KERNEL(sigmoid));
        */
}

pub fn sigmoid_mut(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return ops::activation_(self, VK_KERNEL(sigmoid_));
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("clamp", TORCH_FN(clamp));
      m.impl("clamp_", TORCH_FN(clamp_));
      m.impl("hardsigmoid", hardsigmoid);
      m.impl("hardsigmoid_", hardsigmoid_);
      m.impl("hardswish", hardswish);
      m.impl("hardswish_", hardswish_);
      m.impl("hardtanh", hardtanh);
      m.impl("hardtanh_", hardtanh_);
      m.impl("sigmoid", sigmoid);
      m.impl("sigmoid_", sigmoid_);
      m.impl("relu", relu);
      m.impl("relu_", relu_);
    }
    */
}
