crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Arithmetic.cpp]

pub fn check_inputs(
    input1: &Tensor,
    input2: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          channels_size(input1) == channels_size(input2),
          "Vulkan binary elementwise ops require channel dimension to be equal!");
      if (batch_size(input1) != batch_size(input2)) {
        TORCH_CHECK(
            channels_size(input1) % 4 == 0,
            "Vulkan binary elementwise ops require channel to be a multiple of 4 to broadcast along batch dimension!")
      }

      const u32 input1_h = height_size(input1);
      const u32 input1_w = width_size(input1);
      const u32 input2_h = height_size(input2);
      const u32 input2_w = width_size(input2);

      const string broadcast_error_msg =
          "Incompatible input dimensions for broadcasting for Vulkan binary elementwise op!";
      if (input1_h != input2_h) {
        if (input1_h > input2_h) {
          TORCH_CHECK(input2_h == 1, broadcast_error_msg);
          TORCH_CHECK(input2_w == input1_w || input2_w == 1, broadcast_error_msg);
        } else if (input2_h > input1_h) {
          TORCH_CHECK(input1_h == 1, broadcast_error_msg);
          TORCH_CHECK(input1_w == input2_w || input1_w == 1, broadcast_error_msg);
        }
      } else if (input1_w != input2_w) {
        if (input1_w > input2_w) {
          TORCH_CHECK(input2_w == 1, broadcast_error_msg);
        } else if (input2_w > input1_w) {
          TORCH_CHECK(input1_h == 1, broadcast_error_msg);
        }
      }
        */
}

pub fn broadcast_first_input(
    input1: &VTensor,
    input2: &VTensor) -> bool {
    
    todo!();
        /*
            return (
          (input2.extents().data[1u] > 1 && input1.extents().data[1u] == 1) ||
          (input2.extents().data[2u] > 1 && input1.extents().data[2u] == 1) ||
          input2.extents().data[0u] > input1.extents().data[0u]);
        */
}

pub fn arithmetic_scalar(
    self_arg:          &Tensor,
    other:             &Scalar,
    alpha_arg:         &Option<Scalar>,
    shader_descriptor: &ShaderDescriptor) -> Tensor {
    
    todo!();
        /*
            Context* const context = api::context();

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
        if C10_LIKELY (v_output.has_image() && v_self.has_image()) {
          const float other_val = alpha_arg
              ? other.to<float>() * alpha_arg->to<float>()
              : other.to<float>();
          const struct Block final {
            uvec3 extents;
            float other;
          } block{
              v_self.extents(),
              other_val,
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
                  command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
              // Read-only access is implied on const tensors and triggers an async
              // synchronization if necessary.
              v_self.image(command_buffer, vTensor::Stage::Compute),
              // Object lifetime is managed by the resource pool.
              // It is OK not to keep track of the handle.
              context->resource().pool.uniform(block).object);
        } else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return convert(v_output);
        */
}

pub fn arithmetic_scalar_mut<'a>(
    self_:             &mut Tensor,
    other:             &Scalar,
    alpha_arg:         &Option<Scalar>,
    shader_descriptor: &ShaderDescriptor) -> &'a mut Tensor {
    
    todo!();
        /*
            Context* const context = api::context();

      TORCH_CHECK(
          self.is_vulkan(),
          "Vulkan: In-place add is only supported on Vulkan tensors.");

      vTensor& v_self = convert(self);

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY (v_self.has_image()) {
          const float other_val = alpha_arg
              ? other.to<float>() * alpha_arg->to<float>()
              : other.to<float>();
          const struct Block final {
            uvec3 extents;
            float other;
          } block{
              v_self.extents(),
              other_val,
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
        } else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return self;
        */
}

pub fn arithmetic_tensor(
    self_arg:          &Tensor,
    other_arg:         &Tensor,
    alpha_arg:         &Option<Scalar>,
    shader_descriptor: &ShaderDescriptor) -> Tensor {
    
    todo!();
        /*
            check_inputs(self_arg, other_arg);
      api::Context* const context = api::context();

      const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
      const vTensor& v_self = convert(self);

      const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
      const vTensor& v_other = convert(other);

      vTensor v_output{
          context,
          broadcast_first_input(v_self, v_other) ? v_other.sizes() : v_self.sizes(),
          v_self.options(),
      };

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY (v_self.has_image() && v_other.has_image()) {
          const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0;
          const struct Block final {
            uvec3 extents;
            u32 fill_0;
            uvec3 input1_extents;
            u32 fill_1;
            uvec3 input2_extents;
            float alpha;
          } block{
              v_output.extents(),
              0u,
              v_self.extents(),
              0u,
              v_other.extents(),
              alpha,
          };

          context->dispatch(
              command_buffer,
              {
                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              },
              shader_descriptor,
              v_output.extents(),
              context->gpu().adapter->local_work_group_size(),
              // Write-only access bypasses synchronization but inserts appropriate
              // barriers if necessary.
              v_output.image(
                  command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
              // Read-only access is implied on const tensors and triggers an async
              // synchronization if necessary.
              v_self.image(command_buffer, vTensor::Stage::Compute),
              // Read-only access is implied on const tensors and triggers an async
              // synchronization if necessary.
              v_other.image(command_buffer, vTensor::Stage::Compute),
              // Object lifetime is managed by the resource pool.
              // It is OK not to keep track of the handle.
              context->resource().pool.uniform(block).object);
        } else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return convert(v_output);
        */
}

pub fn arithmetic_tensor_mut<'a>(
    self_:             &mut Tensor,
    other_arg:         &Tensor,
    alpha_arg:         &Option<Scalar>,
    shader_descriptor: &ShaderDescriptor) -> &'a mut Tensor {
    
    todo!();
        /*
            check_inputs(self, other_arg);
      api::Context* const context = api::context();

      TORCH_CHECK(
          self.is_vulkan(),
          "Vulkan: In-place add is only supported on Vulkan tensors.");

      vTensor& v_self = convert(self);

      const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
      const vTensor& v_other = convert(other);

      api::Command::Pool& command_pool = context->command().pool;
      api::Command::Buffer& command_buffer = command_pool.stream();
      {
        if C10_LIKELY (
            v_self.has_image() && v_other.has_image() && !self.is_same(other)) {
          const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0;
          const struct Block final {
            uvec3 extents;
            u32 fill_0;
            uvec3 input_extents;
            float alpha;
          } block{
              v_self.extents(),
              0u,
              v_other.extents(),
              alpha,
          };

          context->dispatch(
              command_buffer,
              {
                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
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
              // Read-only access is implied on const tensors and triggers an async
              // synchronization if necessary.
              v_other.image(command_buffer, vTensor::Stage::Compute),
              // Object lifetime is managed by the resource pool.
              // It is OK not to keep track of the handle.
              context->resource().pool.uniform(block).object);
        } else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return self;
        */
}


pub fn add_scalar(
    self_arg: &Tensor,
    other:    &Scalar,
    alpha:    &Scalar) -> Tensor {
    
    todo!();
        /*
            return arithmetic_scalar(
          self_arg, other, optional<Scalar>(alpha), VK_KERNEL(add_scalar));
        */
}

pub fn add_scalar_mut<'a>(
    self_: &mut Tensor,
    other: &Scalar,
    alpha: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_scalar_(
          self, other, optional<Scalar>(alpha), VK_KERNEL(add_scalar_));
        */
}


pub fn add_tensor(
        self_arg:  &Tensor,
        other_arg: &Tensor,
        alpha:     &Scalar) -> Tensor {
    
    todo!();
        /*
            return arithmetic_tensor(
          self_arg, other_arg, optional<Scalar>(alpha), VK_KERNEL(add));
        */
}

pub fn add_tensor_mut<'a>(
    self_:     &mut Tensor,
    other_arg: &Tensor,
    alpha:     &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_tensor_(
          self, other_arg, optional<Scalar>(alpha), VK_KERNEL(add_));
        */
}

pub fn sub_scalar(
    self_arg: &Tensor,
    other:    &Scalar,
    alpha:    &Scalar) -> Tensor {
    
    todo!();
        /*
            return arithmetic_scalar(
          self_arg,
          other,
          optional<Scalar>(-1 * alpha.to<float>()),
          VK_KERNEL(add_scalar));
        */
}

pub fn sub_scalar_mut<'a>(
    self_: &mut Tensor,
    other: &Scalar,
    alpha: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_scalar_(
          self,
          other,
          optional<Scalar>(-1 * alpha.to<float>()),
          VK_KERNEL(add_scalar_));
        */
}


pub fn sub_tensor(
        self_arg:  &Tensor,
        other_arg: &Tensor,
        alpha:     &Scalar) -> Tensor {
    
    todo!();
        /*
            return arithmetic_tensor(
          self_arg, other_arg, optional<Scalar>(alpha), VK_KERNEL(sub));
        */
}

pub fn sub_tensor_mut<'a>(
        self_:     &mut Tensor,
        other_arg: &Tensor,
        alpha:     &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_tensor_(
          self, other_arg, optional<Scalar>(alpha), VK_KERNEL(sub_));
        */
}


pub fn mul_scalar(
        self_arg: &Tensor,
        other:    &Scalar) -> Tensor {
    
    todo!();
        /*
            return arithmetic_scalar(
          self_arg, other, optional<Scalar>(), VK_KERNEL(mul_scalar));
        */
}

pub fn mul_scalar_mut<'a>(
    self_: &mut Tensor,
    other: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_scalar_(
          self, other, optional<Scalar>(), VK_KERNEL(mul_scalar_));
        */
}


pub fn mul_tensor(
        self_arg:  &Tensor,
        other_arg: &Tensor) -> Tensor {
    
    todo!();
        /*
            return arithmetic_tensor(
          self_arg, other_arg, optional<Scalar>(), VK_KERNEL(mul));
        */
}

pub fn mul_tensor_mut<'a>(
    self_:     &mut Tensor,
    other_arg: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_tensor_(
          self, other_arg, optional<Scalar>(), VK_KERNEL(mul_));
        */
}

pub fn div_scalar(
    self_arg: &Tensor,
    other:    &Scalar) -> Tensor {
    
    todo!();
        /*
            return arithmetic_scalar(
          self_arg,
          1.0 / other.to<float>(),
          optional<Scalar>(),
          VK_KERNEL(mul_scalar));
        */
}

pub fn div_scalar_mut<'a>(
        self_: &mut Tensor,
        other: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_scalar_(
          self,
          1.0 / other.to<float>(),
          optional<Scalar>(),
          VK_KERNEL(mul_scalar_));
        */
}


pub fn div_tensor(
        self_arg:  &Tensor,
        other_arg: &Tensor) -> Tensor {
    
    todo!();
        /*
            return arithmetic_tensor(
          self_arg, other_arg, optional<Scalar>(), VK_KERNEL(div));
        */
}

pub fn div_tensor_mut<'a>(
        self_:     &mut Tensor,
        other_arg: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return arithmetic_tensor_(
          self, other_arg, optional<Scalar>(), VK_KERNEL(div_));
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("add.Scalar", TORCH_FN(add_scalar));
      m.impl("add_.Scalar", TORCH_FN(add_scalar_));
      m.impl("add.Tensor", TORCH_FN(add_tensor));
      m.impl("add_.Tensor", TORCH_FN(add_tensor_));
      m.impl("sub.Scalar", TORCH_FN(sub_scalar));
      m.impl("sub_.Scalar", TORCH_FN(sub_scalar_));
      m.impl("sub.Tensor", TORCH_FN(sub_tensor));
      m.impl("sub_.Tensor", TORCH_FN(sub_tensor_));
      m.impl("mul.Scalar", TORCH_FN(mul_scalar));
      m.impl("mul_.Scalar", TORCH_FN(mul_scalar_));
      m.impl("mul.Tensor", TORCH_FN(mul_tensor));
      m.impl("mul_.Tensor", TORCH_FN(mul_tensor_));
      m.impl("div.Scalar", TORCH_FN(div_scalar));
      m.impl("div_.Scalar", TORCH_FN(div_scalar_));
      m.impl("div.Tensor", TORCH_FN(div_tensor));
      m.impl("div_.Tensor", TORCH_FN(div_tensor_));
    }
    */
}
