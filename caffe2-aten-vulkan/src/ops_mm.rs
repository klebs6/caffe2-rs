// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Mm.h]

pub struct LinearOpContextPacked {
    v_weight: VTensor,
    v_bias:   VTensor,
}

pub struct LinearOpContextUnpacked {
    weight: Tensor,
    bias:   Option<Tensor>,
}

pub struct LinearOpContext {
    base:     CustomClassHolder,
    packed:   LinearOpContextPacked,
    unpacked: LinearOpContextUnpacked,
}

impl HasState for LinearOpContext {

    type State = (Tensor,Option<Tensor>);
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Mm.cpp]

pub fn pack_weights(
        pool:       &mut ResourcePool,
        weight_arg: &Tensor) -> VTensor {
    
    todo!();
        /*
            if (weight_arg.is_vulkan()) {
        return convert(weight_arg);
      }

      Context* const context = context();
      Command::Buffer& command_buffer = context->command().pool.stream();

      const Tensor weight = weight_arg.contiguous();
      const IntArrayRef w_sizes = weight.sizes();
      const float* const src_weight_ptr = weight.data_ptr<float>();

      /* Source */
      const i64 src_kw_sz = w_sizes[Layout::Parameter::width];
      const i64 src_kh_sz = w_sizes[Layout::Parameter::height];

      /* Destination */
      const i64 dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
      const i64 dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
      const i64 dst_plane_sz = dst_kw_sz * dst_kh_sz;

      vTensor v_weight{
          context,
          &pool,
          {
            4,
            dst_kh_sz,
            dst_kw_sz,
          },
          weight.options(),
      };

      using Future = vTensor::Future<float, vTensor::Access::Write>;
      Future v_weight_future = v_weight.host<float, vTensor::Access::Write>(command_buffer);
      Future::Payload v_weight_payload = v_weight_future.wait();

      float* const dst_weight_ptr = v_weight_payload.get();
      memset(dst_weight_ptr, 0, v_weight.nbytes());

      for (i64 src_h = 0; src_h < src_kh_sz; ++src_h) {
        for (i64 src_w = 0; src_w < src_kw_sz; ++src_w) {
          i64 dst_plane = 2*(src_h%2) + (src_w%2);
          i64 dst_index = (src_h/2)*dst_kw_sz + (src_w/2);
          memcpy(
              dst_weight_ptr + dst_plane * dst_plane_sz + dst_index,
              src_weight_ptr + src_h * src_kw_sz + src_w,
              sizeof(float));
        }
      }

      return v_weight;
        */
}



pub fn pack_biases(
        pool:       &mut ResourcePool,
        weight_arg: &Tensor,
        bias_arg:   &Option<Tensor>) -> VTensor {
    
    todo!();
        /*
            if (bias_arg && bias_arg->is_vulkan()) {
        return convert(*bias_arg);
      }

      Context* const context = context();
      Command::Buffer& command_buffer = context->command().pool.stream();

      using Future = vTensor::Future<float, vTensor::Access::Write>;
      if (bias_arg) {
        const Tensor bias = bias_arg->contiguous();
        const IntArrayRef b_sizes = bias.sizes();
        const float* const src_bias_ptr = bias.data_ptr<float>();

        /* Source */
        i64 src_kw_sz, src_kh_sz;
        if (bias.sizes().size() == 2) {
          src_kw_sz = b_sizes[Layout::Parameter::width];
          src_kh_sz = b_sizes[Layout::Parameter::height];
        }
        else {
          src_kw_sz = b_sizes[Layout::Parameter::height];
          src_kh_sz = 1;
        }

        /* Destination */
        const i64 dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
        const i64 dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
        const i64 dst_plane_sz = dst_kw_sz * dst_kh_sz;

        vTensor v_bias{
            context,
            &pool,
            {
              4,
              dst_kh_sz,
              dst_kw_sz,
            },
            bias_arg->options(),
        };

        Future v_bias_future = v_bias.host<float, vTensor::Access::Write>(command_buffer);
        Future::Payload v_bias_payload = v_bias_future.wait();

        float* const dst_bias_ptr = v_bias_payload.get();
        memset(dst_bias_ptr, 0, v_bias.nbytes());

        for (i64 src_h = 0; src_h < src_kh_sz; ++src_h) {
          for (i64 src_w = 0; src_w < src_kw_sz; ++src_w) {
            i64 dst_plane = 2*(src_h%2) + (src_w%2);
            i64 dst_index = (src_h/2)*dst_kw_sz + (src_w/2);
            memcpy(
                dst_bias_ptr + dst_plane * dst_plane_sz + dst_index,
                src_bias_ptr + src_h * src_kw_sz + src_w,
                sizeof(float));
          }
        }

        return v_bias;
      }
      else {
        vTensor v_bias{
            context(),
            &pool,
            {1},
            weight_arg.options(),
        };
        Future v_bias_future = v_bias.host<float, vTensor::Access::Write>(command_buffer);
        Future::Payload v_bias_payload = v_bias_future.wait();
        memset(
            v_bias_payload.get(),
            // 2's complement integers and IEEE-754 floating point numbers both
            // have identical bit representations for 0, so can use memset which
            // only accepts u8 parameter.
            0,
            v_bias.nbytes());

        return v_bias;
      }
        */
}


pub fn available(
        weight: &Tensor,
        bias:   &Option<Tensor>) -> bool {
    
    todo!();
        /*
            return available() &&
             // Weight
             (2 == weight.ndimension()) &&
             (weight.size(Layout::Parameter::height) > 0) &&
             (weight.size(Layout::Parameter::width) > 0) &&
             ((weight.device().is_cpu()) ||
              (DeviceType_Vulkan == weight.device().type())) &&
             (kFloat == weight.scalar_type()) &&
             !weight.requires_grad() &&
             // Bias
             ((bias && bias->defined()) ? ((bias->ndimension() > 0) &&
                                           ((bias->device().is_cpu()) ||
                                            (DeviceType_Vulkan == bias->device().type())) &&
                                           (kFloat == bias->scalar_type()) &&
                                           ((bias->ndimension() > 1) ?
                                                (bias->size(Layout::Parameter::width) ==
                                                    weight.size(Layout::Parameter::width))
                                                : true) &&
                                           !bias->requires_grad())
                                        : true) &&
             true;
        */
}



pub fn usable(
        input:  &Tensor,
        weight: &Tensor,
        bias:   &Option<Tensor>) -> bool {
    
    todo!();
        /*
            return (2 == input.ndimension()) &&
             (DeviceType_Vulkan == input.device().type()) &&
             (kFloat == input.scalar_type()) &&
             (input.size(Layout::Parameter::width) ==
                  weight.size(Layout::Parameter::height)) &&
             !input.requires_grad() &&
             true;
        */
}


pub fn addmm(
        bias:   &Tensor,
        input:  &Tensor,
        weight: &Tensor,
        beta:   &Scalar,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            return LinearOpContext::create(
          context()->resource().pool,
          weight,
          bias).run(
              input,
              alpha.to<float>(),
              beta.to<float>());
        */
}


pub fn mm(
        mat1_arg: &Tensor,
        mat2_arg: &Tensor) -> Tensor {
    
    todo!();
        /*
            return LinearOpContext::create(
          context()->resource().pool,
          mat2_arg,
          optional<Tensor>()).run(
              mat1_arg,
              1.0f,
              1.0f);
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("addmm", TORCH_FN(addmm));
      m.impl("mm", TORCH_FN(mm));
    }
    */
}

impl LinearOpContext {
    
    pub fn new(
        pool:   &mut ResourcePool,
        weight: &Tensor,
        bias:   &Option<Tensor>) -> Self {
    
        todo!();
        /*


            : packed_{
          pack_weights(pool, weight),
          pack_biases(pool, weight, bias),
        },
        unpacked_{
          weight,
          bias,
        }
        */
    }
    
    pub fn create(&mut self, 
        pool:   &mut ResourcePool,
        weight: &Tensor,
        bias:   &Option<Tensor>) -> LinearOpContext {
        
        todo!();
        /*
            TORCH_CHECK(
          available(weight, bias),
          "Vulkan Linear not available! "
          "Reason: The provided (weight, bias) parameters are either invalid "
          "individually or their combination is not supported by Vulkan Impl.");

      // Pass in the originals
      return LinearOpContext{
          pool,
          weight,
          bias,
      };
        */
    }
    
    pub fn run(&self, 
        input_arg: &Tensor,
        alpha:     f32,
        beta:      f32) -> Tensor {
        
        todo!();
        /*
            Context* const context = context();

      const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
      const vTensor& v_input = convert(input);

      TORCH_CHECK(
          usable(input, unpacked_.weight, unpacked_.bias),
          "Vulkan Linear not usable! "
          "Reason: The provided input tensor is either invalid on its own, or its "
          "combination with the provided weight and bias tensors are unsupported by "
          "Vulkan impl.");

      SmallVector<i64, 4u> output_sizes{
          v_input.sizes()[Layout::Parameter::height],
          unpacked_.weight.sizes()[Layout::Parameter::width],
      };

      vTensor v_output {
          context,
          {
            v_input.sizes()[Layout::Parameter::height],
            unpacked_.weight.sizes()[Layout::Parameter::width],
          },
          input.options(),
      };

      Command::Pool& command_pool = context->command().pool;

      Command::Buffer& command_buffer = command_pool.stream();
      {
        if (v_input.has_image() &&
            packed_.v_weight.has_image() &&
            packed_.v_bias.has_image()) {
          if (unpacked_.bias && unpacked_.bias->defined()) {
            const struct {
              uvec3 size;
              i32 K;
              vec2 multiplier;
            } block {
                v_output.extents(),
                safe_downcast<i32>(div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
                {
                  alpha,
                  beta,
                },
            };

            context->dispatch(
                command_buffer,
                {
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                },
                VK_KERNEL(addmm),
                {
                  safe_downcast<u32>(div_up(unpacked_.weight.sizes()[Layout::Parameter::width], INT64_C(2))),
                  safe_downcast<u32>(div_up(v_input.sizes()[Layout::Parameter::height], INT64_C(2))),
                  1,
                },
                {8, 8, 1},
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
                // Read-only access is implied on const tensors and triggers an async
                // synchronization if necessary.
                packed_.v_weight.image(
                    command_buffer,
                    vTensor::Stage::Compute),
                // Read-only access is implied on const tensors and triggers an async
                // synchronization if necessary.
                packed_.v_bias.image(
                    command_buffer,
                    vTensor::Stage::Compute),
                // Object lifetime is managed by the resource pool.
                // It is OK not to keep track of the handle.
                context->resource().pool.uniform(block).object);
          }
          else {
            const struct {
              uvec3 size;
              i32 K;
            } block_no_bias {
                v_output.extents(),
                safe_downcast<i32>(div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
            };

            context->dispatch(
                command_buffer,
                {
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                },
                VK_KERNEL(mm),
                {
                  safe_downcast<u32>(div_up(unpacked_.weight.sizes()[Layout::Parameter::width], INT64_C(2))),
                  safe_downcast<u32>(div_up(v_input.sizes()[Layout::Parameter::height], INT64_C(2))),
                  1,
                },
                {8, 8, 1},
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
                // Read-only access is implied on const tensors and triggers an async
                // synchronization if necessary.
                packed_.v_weight.image(
                    command_buffer,
                    vTensor::Stage::Compute),
                // Object lifetime is managed by the resource pool.
                // It is OK not to keep track of the handle.
                context->resource().pool.uniform(block_no_bias).object);
          }
        }
        else {
          TORCH_CHECK(false, "Not implemented!");
        }
      }
      command_pool.submit(context->gpu().queue, command_buffer);

      return convert(v_output);
        */
    }
    
    pub fn unpack(&self) -> LinearOpContextState {
        
        todo!();
        /*
            return LinearOpContext::State{
          unpacked_.weight,
          unpacked_.bias,
      };
        */
    }
}

pub fn linear_prepack(
        weight: Tensor,
        bias:   Option<Tensor>) -> IntrusivePtr<LinearOpContext> {
    
    todo!();
        /*
            return make_intrusive<LinearOpContext>(
          LinearOpContext::create(
              persistent()->pool,
              move(weight),
              move(bias)));
        */
}

pub fn linear_run(
        input:   &Tensor,
        context: &IntrusivePtr<LinearOpContext>) -> Tensor {
    
    todo!();
        /*
            return context->run(input, 1.0, 1.0);
        */
}
