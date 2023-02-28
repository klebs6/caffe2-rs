// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanOps.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanOps.cpp]

pub fn upsample_nearest2d(
    output: &mut VulkanTensor,
    input:  &VulkanTensor,
    IH:     i64,
    IW:     i64,
    OH:     i64,
    OW:     i64,
    IN:     i64,
    IC:     i64,
    scaleh: f32,
    scalew: f32)  {

    todo!();
        /*
            auto device = context().device();
      i64 C = IN * IC;
      struct ConstBlock {
        float scaleX;
        float scaleY;
      };
      ConstBlock cb{scaleW,
                    scaleH};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(upsample_nearest2d), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
      computeUnit.dispatchCommandBuffer(OW, OH, C, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}

pub fn reshape_copy(
    input: &VulkanTensor,
    shape: Vec<i64>) -> VulkanTensor {

    todo!();
        /*
            input.sync_image_to_buffer();
      VulkanTensor output{infer_size(shape, input.numel())};
      copy_buffer_to_buffer(
          *(input.buffer()), *(output.buffer()), input.buffer()->sizeBytes());
      return output;
        */
}

pub fn cat(
    output: &mut VulkanTensor,
    inputs: &[VulkanTensor],
    dim:    i64) -> VulkanTensor {
    
    todo!();
        /*
            VkDeviceSize outputOffset = 0;
      for (const auto& input : inputs) {
        input.sync_image_to_buffer();
        const auto sizeBytes = sizeof(float) * input.numel();
        copy_buffer_to_buffer(
            *(input.buffer()), *(output.buffer()), sizeBytes, 0, outputOffset);
        outputOffset += sizeBytes;
      }
      return output;
        */
}

pub fn adaptive_avg_pool2d(
    output: &mut VulkanTensor,
    input:  &VulkanTensor,
    IH:     i64,
    IW:     i64,
    OH:     i64,
    OW:     i64,
    IN:     i64,
    IC:     i64)  {

    todo!();
        /*
            auto device = context().device();
      i64 C = IN * IC;

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(adaptive_avg_pool2d), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
      computeUnit.dispatchCommandBuffer(OW, OH, C, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}

pub fn max_pool2d(
    output:    &mut VulkanTensor,
    input:     &VulkanTensor,
    ih:        i32,
    iw:        i32,
    oh:        i32,
    ow:        i32,
    n:         i32,
    c:         i32,
    kh:        i32,
    kw:        i32,
    dh:        i32,
    dw:        i32,
    padh:      i32,
    padw:      i32,
    dilationh: i32,
    dilationw: i32)  {
    
    todo!();
        /*
            auto device = context().device();
      const auto c = _n * _c;
      struct ConstBlock {
        i32 inputSize[4];
        i32 outputSize[4];
        i32 kernelSize[2];
        i32 stride[2];
        i32 padding[2];
        i32 dilate[2];
      };
      ConstBlock cb{
          {iW, iH, c, 0},
          {oW, oH, c, 0},
          {kW, kH},
          {dW, dH},
          {padW, padH},
          {dilationW, dilationH},
      };
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(max_pool2d), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
      computeUnit.dispatchCommandBuffer(oW, oH, c, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}

pub fn avg_pool2d(
    output: &mut VulkanTensor,
    input:  &VulkanTensor,
    ih:     i32,
    iw:     i32,
    oh:     i32,
    ow:     i32,
    n:      i32,
    c:      i32,
    kh:     i32,
    kw:     i32,
    dh:     i32,
    dw:     i32,
    padh:   i32,
    padw:   i32)  {

    todo!();
    /*
            auto device = context().device();
      const auto c = _n * _c;
      struct ConstBlock {
        i32 kernelSize[2];
        i32 stride[2];
        i32 padding[2];
      };
      ConstBlock cb{
          {kW, kH},
          {dW, dH},
          {padW, padH},
      };
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(avg_pool2d), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      input.image()->addImageMemoryBarrierToShaderRead(computeUnit.commandBuffer());
      computeUnit.dispatchCommandBuffer(oW, oH, c, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}


pub fn transpose(
        input: &VulkanTensor,
        dim0:  i64,
        dim1:  i64) -> VulkanTensor {
    
    todo!();
        /*
            const auto idim = input.dim();
      TORCH_INTERNAL_ASSERT(
          idim <= 6, "Vulkan transpose is implemented only for dim <= 6");
      auto device = context().device();
      struct ConstBlock {
        i32 istrides[8];
        i32 ostrides[8];
        i32 odims[8];
        i32 storageOffset;
      };

      auto isizes = input.sizes();
      auto osizes = isizes;
      swap(osizes[dim0], osizes[dim1]);
      VulkanTensor output{osizes};
      output.allocate_storage();

      array<i32, 8> idims8;
      idims8.fill(1);
      array<i32, 8> odims8;
      odims8.fill(1);
      copy(isizes.cbegin(), isizes.cend(), idims8.end() - idim);
      copy(osizes.cbegin(), osizes.cend(), odims8.end() - idim);
      array<i32, 8> istrides8;
      istrides8.fill(1);
      array<i32, 8> ostrides8;
      ostrides8.fill(1);
      for (int i = 6; i >= 0; --i) {
        istrides8[i] = idims8[i + 1] * istrides8[i + 1];
        ostrides8[i] = odims8[i + 1] * ostrides8[i + 1];
      }
      swap(istrides8[8 - idim + dim0], istrides8[8 - idim + dim1]);

      ConstBlock cb{};
      copy(istrides8.cbegin(), istrides8.cend(), begin(cb.istrides));
      copy(ostrides8.cbegin(), ostrides8.cend(), begin(cb.ostrides));
      copy(odims8.cbegin(), odims8.cend(), begin(cb.odims));
      cb.storageOffset = 0;

      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.buffer()->bind(descriptorSet, 0);
      input.buffer()->bind(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(permute), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      input.buffer()->addBufferMemoryBarrier(
          computeUnit.commandBuffer(), 0, input.buffer()->sizeBytes());
      computeUnit.dispatchCommandBuffer(
          odims8[6] * odims8[7],
          odims8[4] * odims8[5],
          odims8[2] * odims8[3],
          workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
      return output;
        */
}


pub fn slice(
        input: &VulkanTensor,
        dim:   i64,
        start: i64,
        end:   i64,
        step:  i64) -> VulkanTensor {
    
    todo!();
        /*
            const auto isizes = input.sizes();
      auto osizes = isizes;
      auto start = _start;
      auto end = _end;
      if (start < 0) {
        start += isizes[dim];
      }
      if (end < 0) {
        end += isizes[dim];
      }
      if (start < 0) {
        start = 0;
      } else if (start >= isizes[dim]) {
        start = isizes[dim];
      }
      if (end < start) {
        end = start;
      } else if (end >= isizes[dim]) {
        end = isizes[dim];
      }
      const auto len = end - start;
      osizes[dim] = (len + step - 1) / step;

      VulkanTensor output{osizes};
      output.allocate_storage();

      auto idim = input.dim();
      array<i32, 8> idims8;
      idims8.fill(1);
      copy(isizes.cbegin(), isizes.cend(), idims8.end() - idim);
      array<i32, 8> istrides8;
      istrides8.fill(1);
      for (int i = 6; i >= 0; --i) {
        istrides8[i] = idims8[i + 1] * istrides8[i + 1];
      }

      array<i32, 8> odims8 = idims8;
      array<i32, 8> ostrides8 = istrides8;

      ostrides8[8 - idim + dim] *= step;
      auto storage_offset = start * istrides8[8 - idim + dim];

      auto device = context().device();
      struct ConstBlock {
        i32 istrides[8];
        i32 ostrides[8];
        i32 odims[8];
        i32 storageOffset;
      };

      ConstBlock cb{};
      copy(istrides8.cbegin(), istrides8.cend(), begin(cb.istrides));
      copy(ostrides8.cbegin(), ostrides8.cend(), begin(cb.ostrides));
      copy(odims8.cbegin(), odims8.cend(), begin(cb.odims));
      cb.storageOffset = storage_offset;

      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.buffer()->bind(descriptorSet, 0);
      input.buffer()->bind(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(permute), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      input.buffer()->addBufferMemoryBarrier(
          computeUnit.commandBuffer(), 0, input.buffer()->sizeBytes());
      computeUnit.dispatchCommandBuffer(
          odims8[6] * odims8[7],
          odims8[4] * odims8[5],
          odims8[2] * odims8[3],
          workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
      return output;
        */
}

pub fn add_a(
        output: &mut VulkanTensor,
        input0: &VulkanTensor,
        input1: &VulkanTensor,
        alpha:  f32)  {
    
    todo!();
        /*
            auto odim = output.dim();
      TORCH_INTERNAL_ASSERT(
          odim <= 4, "Vulkan add is implemented for dim <= 4, output dim > 4");
      auto i0dim = input0.dim();
      TORCH_INTERNAL_ASSERT(
          i0dim <= 4, "Vulkan add is implemented for dim <= 4, input0 dim > 4");
      auto i1dim = input1.dim();
      TORCH_INTERNAL_ASSERT(
          i1dim <= 4, "Vulkan add is implemented for dim <= 4, input1 dim > 4");

      auto os = output.sizes();
      auto i0s = input0.sizes();
      auto i1s = input1.sizes();

      array<i64, 4> os4 = {1, 1, 1, 1};
      copy(os.begin(), os.end(), os4.end() - odim);
      array<i64, 4> i0s4 = {1, 1, 1, 1};
      copy(i0s.cbegin(), i0s.cend(), i0s4.end() - i0dim);
      array<i64, 4> i1s4 = {1, 1, 1, 1};
      copy(i1s.cbegin(), i1s.cend(), i1s4.end() - i1dim);

      TORCH_INTERNAL_ASSERT(
          (os4 == i0s4) && (i0s4 == i1s4),
          "Vulkan add expects the same dimensions for all operands");

      auto C = os4[0] * os4[1];
      auto H = os4[2];
      auto W = os4[3];

      auto device = context().device();
      struct ConstBlock {
        float alpha;
      };
      ConstBlock cb{alpha};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input0.image()->bindShaderRead(descriptorSet, 1);
      input1.image()->bindShaderRead(descriptorSet, 2);
      constBuffer.bind(descriptorSet, 3);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(add), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
      input0.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      input1.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      computeUnit.dispatchCommandBuffer(W, H, C, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}

pub fn add_b(
    output: &mut VulkanTensor,
    input:  &VulkanTensor,
    s:      f32)  {
    
    todo!();
        /*
            const auto sizes = input.sizes();

      const auto C = multiply_integers(sizes.cbegin(), sizes.cend() - 2);
      const auto C_4 = UP_DIV(C, 4);
      const auto H = sizes[2];
      const auto W = sizes[3];

      auto device = context().device();
      struct ConstBlock {
        float s;
      };
      ConstBlock cb{s};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(add_scalar), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
      input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      computeUnit.dispatchCommandBuffer(W, H, C_4, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}

pub fn mul(
    output: &mut VulkanTensor,
    input:  &VulkanTensor,
    s:      f32)  {
    
    todo!();
        /*
            const auto sizes = input.sizes();

      const auto C = multiply_integers(sizes.cbegin(), sizes.cend() - 2);
      const auto C_4 = UP_DIV(C, 4);
      const auto H = sizes[2];
      const auto W = sizes[3];

      auto device = context().device();
      struct ConstBlock {
        float s;
      };
      ConstBlock cb{s};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(mul_scalar), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
      input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      computeUnit.dispatchCommandBuffer(W, H, C_4, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}

pub fn kernelnchw_ochw_repack_o4c4h_wi4o4(
    weights: *const f32,
    OC:      i32,
    C:       i32,
    KH:      i32,
    KW:      i32) -> VBuffer {

    todo!();
        /*
            const auto C_4 = UP_DIV(C, 4);
      const auto kBufSizeNumel = ALIGN_UP4(OC) * ALIGN_UP4(C) * KH * KW;
      auto size = sizeof(float) * kBufSizeNumel;
      VBuffer kernelBuffer{size};
      const int oc_4SizeNumel = KW * KH * C_4 * 16;
      auto mappedMemory = kernelBuffer.map();
      if (mappedMemory.ptr()) {
        float* basePtr = (float*)mappedMemory.ptr();
        memset(basePtr, 0, size);
        const float* src = weights;
        int ridx = 0;
        for (int oc = 0; oc < OC; ++oc) {
          int oc_4 = oc / 4;
          int oc_4_i = oc % 4;
          float* dst_oc = basePtr + oc_4 * oc_4SizeNumel;
          for (int ic = 0; ic < C; ++ic) {
            int ic_4 = ic / 4;
            int ic_4_i = ic % 4;
            float* dst_ic = dst_oc + ic_4 * KW * KH * 16;
            for (int ky = 0; ky < KH; ++ky) {
              float* dst_ky = dst_ic + ky * KW * 16;
              for (int kx = 0; kx < KW; ++kx) {
                float* dst_kx = dst_ky + kx * 16;
                dst_kx[4 * ic_4_i + oc_4_i] = src[ridx++];
              }
            }
          }
        }
      }
      mappedMemory.flushWriteToDevice();
      return kernelBuffer;
        */
}

pub fn buffer_from_optional_host_data(
    data:        Option<*const f32>,
    data_size:   u32,
    buffer_size: u32) -> VBuffer {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          dataSize <= bufferSize,
          "buffer size(",
          bufferSize,
          ") is not enough for data(",
          dataSize,
          ")");
      const auto sizeAligned =
          ROUND_UP(bufferSize, context().limits().minStorageBufferOffsetAlignment);
      VBuffer buffer{sizeAligned};
      if (data.has_value()) {
        buffer.copy_from_host_to_device(*data, dataSize);
      } else {
        buffer.set_zeros();
      }
      return buffer;
        */
}

pub fn buffer_zeros(size: u32) -> VBuffer {
    
    todo!();
        /*
            VBuffer buffer{size};
      buffer.set_zeros();
      return buffer;
        */
}

pub fn conv2d_depthwise_a(
    output:      &mut VulkanTensor,
    input:       &VulkanTensor,
    weight:      &VulkanTensor,
    bias_buffer: &VBuffer,
    params:      &Conv2DParams,
    output_min:  Option<f32>,
    output_max:  Option<f32>)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(params.G == params.C);
      auto osizes = output.sizes();
      TORCH_INTERNAL_ASSERT(osizes[2] == params.OH);
      TORCH_INTERNAL_ASSERT(osizes[3] == params.OW);
      struct ConstBlock {
        i32 padding[2];
        i32 kernelSize[2];
        i32 stride[2];
        i32 dilate[2];
        i32 inputSize[4];
        i32 outputSize[4];
        float outputMin;
        float outputMax;
      };
      ConstBlock cb{
          {safe_downcast<i32>(params.PX), safe_downcast<i32>(params.PY)},
          {safe_downcast<i32>(params.KW), safe_downcast<i32>(params.KH)},
          {safe_downcast<i32>(params.SX), safe_downcast<i32>(params.SY)},
          {safe_downcast<i32>(params.DX), safe_downcast<i32>(params.DY)},
          {safe_downcast<i32>(params.OW),
           safe_downcast<i32>(params.OH),
           safe_downcast<i32>(params.OC_4),
           0},
          {safe_downcast<i32>(params.W),
           safe_downcast<i32>(params.H),
           safe_downcast<i32>(params.C_4),
           0},
          output_min ? *output_min : -numeric_limits<float>::infinity(),
          output_max ? *output_max : numeric_limits<float>::infinity()};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      auto device = context().device();
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      weight.image()->bindShaderRead(descriptorSet, 2);
      biasBuffer.bind(descriptorSet, 3);
      constBuffer.bind(descriptorSet, 4);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(conv2d_dw_clamp), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
      input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      weight.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      computeUnit.dispatchCommandBuffer(
          params.OW, params.OH, params.OC_4, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();

      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}

pub fn conv2d_depthwise_b(
    output:     &mut VulkanTensor,
    input:      &VulkanTensor,
    weight:     &VulkanTensor,
    bias:       Option<*const f32>,
    params:     Conv2DParams,
    output_min: Option<f32>,
    output_max: Option<f32>)  {

    todo!();
        /*
            conv2d_depthwise(
          output,
          input,
          weight,
          bufferFromOptionalHostData(
              bias,
              sizeof(float) * params.OC,
              sizeof(float) * ALIGN_UP4(params.OC)),
          params,
          output_min,
          output_max);
        */
}

pub fn conv2d_depthwise_c(
    output:     &mut VulkanTensor,
    input:      &VulkanTensor,
    weight:     *const f32,
    bias:       Option<*const f32>,
    params:     Conv2DParams,
    output_min: Option<f32>,
    output_max: Option<f32>)  {

    todo!();
        /*
            VulkanTensor weightTensor{{params.OC, params.KH, params.KW}};
      weightTensor.set_data_from_host(weight);
      conv2d_depthwise(
          output,
          input,
          weightTensor,
          bufferFromOptionalHostData(
              bias,
              sizeof(float) * params.OC,
              sizeof(float) * ALIGN_UP4(params.OC)),
          params,
          output_min,
          output_max);
        */
}


pub fn conv2d_prepack_weights_image_sizes(
        argoc: i64,
        argc:  i64,
        KH:    i64,
        KW:    i64) -> ImageSizes {
    
    todo!();
        /*
            const i32 C = safe_downcast<i32>(argC);
      const i32 OC = safe_downcast<i32>(argOC);
      const i32 Cup4 = ALIGN_UP4(C);
      const i32 OC_4 = UP_DIV(OC, 4);
      const i32 Z = safe_downcast<i32>(KH) * safe_downcast<i32>(KW);
      return {{Cup4, OC_4, Z}, {Cup4, OC_4, Z}};
        */
}


pub fn conv2d_prepack_weights_to_image(
        image:  &mut VImage,
        weight: *const f32,
        OC:     i64,
        C:      i64,
        KH:     i64,
        KW:     i64)  {
    
    todo!();
        /*
            auto kernelBuffer = kernelNCHW_OCHW_repack_O4C4HWi4o4(weight, OC, C, KH, KW);
      auto OC_4 = UP_DIV(OC, 4);
      auto C_4 = UP_DIV(C, 4);

      auto expectedSizes = conv2d_prepack_weights_image_sizes(OC, C, KH, KW);
      TORCH_INTERNAL_ASSERT(
          image.sizes() == expectedSizes.imageSize,
          "Out VImage sizes do not match expected");

      struct ConstBlock {
        i32 KWxKH;
        i32 C_4;
      };
      ConstBlock cb{safe_downcast<i32>(KW * KH), safe_downcast<i32>(C_4)};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          context().device(),
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      image.bindStorageImage(descriptorSet, 0);
      kernelBuffer.bind(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{1, 1, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(KO4C4HW_to_image), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      image.addImageMemoryBarrierToGeneral(commandBuffer);
      kernelBuffer.addBufferMemoryBarrier(
          commandBuffer, 0, kernelBuffer.sizeBytes());
      computeUnit.addMemoryBarrier(
          VK_PIPELINE_STAGE_HOST_BIT,
          VK_ACCESS_HOST_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);
      computeUnit.dispatchCommandBuffer(C_4, OC_4, KH * KW, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(context().device(), descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(
          context().device(), descriptorSetLayout, nullptr);
        */
}


pub fn conv2d_prepack_weights_image(
        weight: *const f32,
        OC:     i64,
        C:      i64,
        KH:     i64,
        KW:     i64) -> VImage {
    
    todo!();
        /*
            VImage image{conv2d_prepack_weights_image_sizes(OC, C, KH, KW)};
      conv2d_prepack_weights_to_image(image, weight, OC, C, KH, KW);
      return image;
        */
}


pub fn conv2d_prepack_weights(
        output: &mut VulkanTensor,
        weight: *const f32,
        OC:     i64,
        C:      i64,
        KH:     i64,
        KW:     i64)  {
    
    todo!();
        /*
            auto imageSizes = conv2d_prepack_weights_image_sizes(OC, C, KH, KW);
      conv2d_prepack_weights_to_image(
          *(output.image(imageSizes)), weight, OC, C, KH, KW);
        */
}

pub fn conv2d_a(
        output:       &mut VulkanTensor,
        input:        &VulkanTensor,
        kernel_image: &VImage,
        bias_buffer:  &VBuffer,
        params:       &Conv2DParams,
        output_min:   Option<f32>,
        output_max:   Option<f32>)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          params.G == 1, "Prepacked kernel VImage for non-group conv2d only");
      auto osizes = output.sizes();
      TORCH_INTERNAL_ASSERT(
          osizes[2] == params.OH,
          "Output tensor dims do not match specified conv2d params");
      TORCH_INTERNAL_ASSERT(
          osizes[3] == params.OW,
          "Output tensor dims do not match specified conv2d params");

      struct ConstBlock {
        i32 padding[2];
        i32 kernelSize[2];
        i32 stride[2];
        i32 dilate[2];
        i32 inputSize[4];
        i32 outputSize[4];
        float outputMin;
        float outputMax;
      };
      float outputMin =
          output_min ? *output_min : -numeric_limits<float>::infinity();
      float outputMax =
          output_max ? *output_max : numeric_limits<float>::infinity();
      ConstBlock cb{
          {safe_downcast<i32>(params.PX), safe_downcast<i32>(params.PY)},
          {safe_downcast<i32>(params.KW), safe_downcast<i32>(params.KH)},
          {safe_downcast<i32>(params.SX), safe_downcast<i32>(params.SY)},
          {safe_downcast<i32>(params.DX), safe_downcast<i32>(params.DY)},
          {safe_downcast<i32>(params.OW),
           safe_downcast<i32>(params.OH),
           safe_downcast<i32>(params.OC_4),
           safe_downcast<i32>(params.OC)},
          {safe_downcast<i32>(params.W),
           safe_downcast<i32>(params.H),
           safe_downcast<i32>(params.C_4),
           safe_downcast<i32>(params.C)},
          outputMin,
          outputMax};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      auto device = context().device();
      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      kernelImage.bindShaderRead(descriptorSet, 2);
      biasBuffer.bind(descriptorSet, 3);
      constBuffer.bind(descriptorSet, 4);

      WorkGroupSize workGroupSize{1, 1, safe_downcast<u32>(params.OC_4)};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(conv2d_nogroup_clamp), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
      input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      kernelImage.addImageMemoryBarrierToShaderRead(commandBuffer);
      computeUnit.dispatchCommandBuffer(
          UP_DIV(params.OW, 4 * workGroupSize.x),
          UP_DIV(params.OH, workGroupSize.y),
          UP_DIV(params.OC_4, workGroupSize.z));
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();

      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}


pub fn conv2d_b(
        output:       &mut VulkanTensor,
        input:        &VulkanTensor,
        kernel_image: &VImage,
        bias:         Option<*const f32>,
        params:       &Conv2DParams,
        output_min:   Option<f32>,
        output_max:   Option<f32>)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          params.G == 1, "Prepacked kernel VImage for non-group conv2d only");
      conv2d(
          output,
          input,
          kernelImage,
          bufferFromOptionalHostData(
              bias,
              sizeof(float) * params.OC,
              sizeof(float) * ALIGN_UP4(params.OC)),
          params,
          output_min,
          output_max);
        */
}


pub fn conv2d_c(
        output:           &mut VulkanTensor,
        input:            &VulkanTensor,
        weight_prepacked: &VulkanTensor,
        bias:             Option<*const f32>,
        params:           Conv2DParams,
        output_min:       Option<f32>,
        output_max:       Option<f32>)  {
    
    todo!();
        /*
            if (params.G > 1) {
        conv2d_depthwise(
            output,
            input,
            weight_prepacked,
            bufferFromOptionalHostData(
                bias,
                sizeof(float) * params.OC,
                sizeof(float) * ALIGN_UP4(params.OC)),
            params,
            output_min,
            output_max);
        return;
      }

      conv2d(
          output,
          input,
          *(weight_prepacked.image()),
          bias,
          params,
          output_min,
          output_max);
        */
}

pub fn conv2d_d(
        output:           &mut VulkanTensor,
        input:            &VulkanTensor,
        weight_prepacked: &VulkanTensor,
        bias:             &VulkanTensor,
        params:           Conv2DParams,
        output_min:       Option<f32>,
        output_max:       Option<f32>)  {
    
    todo!();
        /*
            if (params.G > 1) {
        conv2d_depthwise(
            output,
            input,
            weight_prepacked,
            *(bias.buffer()),
            params,
            output_min,
            output_max);
        return;
      }

      conv2d(
          output,
          input,
          *(weight_prepacked.image()),
          *(bias.buffer()),
          params,
          output_min,
          output_max);
        */
}

pub fn conv2d_e(
    output:     &mut VulkanTensor,
    input:      &VulkanTensor,
    weight:     *const f32,
    bias:       Option<*const f32>,
    params:     Conv2DParams,
    output_min: Option<f32>,
    output_max: Option<f32>)  {

    todo!();
    /*
      if (params.G > 1) {
        TORCH_INTERNAL_ASSERT(
            params.G == params.C,
            "Vulkan conv2d supports only no-group and depthwise");
        conv2d_depthwise(
            output, input, weight, bias, params, output_min, output_max);
        return;
      }

      conv2d(
          output,
          input,
          conv2d_prepack_weights_image(
              weight, params.OC, params.C, params.KH, params.KW),
          bias,
          params,
          output_min,
          output_max);
        */
}


pub fn clamp(
        output: &mut VulkanTensor,
        input:  &VulkanTensor,
        min:    f32,
        max:    f32)  {
    
    todo!();
        /*
            auto sizes = output.sizes();
      auto C = sizes[0] * sizes[1];
      auto H = sizes[2];
      auto W = sizes[3];

      auto device = context().device();
      struct ConstBlock {
        float min;
        float max;
      };
      ConstBlock cb{min, max};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{8, 8, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(clamp), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
      input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      computeUnit.dispatchCommandBuffer(W, H, C, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}


pub fn addmm(
        output: &mut VulkanTensor,
        t:      Option<VulkanTensor>,
        m1:     &VulkanTensor,
        m2:     &VulkanTensor,
        beta:   f32,
        alpha:  f32)  {
    
    todo!();
        /*
            bool hasT = t.has_value();
      const auto m1Sizes = m1.sizes();
      const auto m2Sizes = m2.sizes();
      TORCH_INTERNAL_ASSERT(m1Sizes.size() == 2);
      TORCH_INTERNAL_ASSERT(m2Sizes.size() == 2);
      const auto m1W = m1Sizes[1];
      const auto m1C = 1;
      const auto m2H = m2Sizes[0];
      const auto m2C = 1;
      const auto OH = m1Sizes[0];
      const auto OW = m2Sizes[1];

      TORCH_INTERNAL_ASSERT(m1W == m2H);
      TORCH_INTERNAL_ASSERT(m1C == m2C);

      const auto C = m1C;
      const auto C_4 = UP_DIV(C, 4);

      auto device = context().device();

      struct ConstBlock {
        float alpha;
        float beta;
      };
      ConstBlock cb{alpha, beta};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{};
      if (hasT) {
        descriptorTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
      } else {
        descriptorTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        };
      }

      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      m1.image()->bindShaderRead(descriptorSet, 1);
      m2.image()->bindShaderRead(descriptorSet, 2);
      if (hasT) {
        (*t).image()->bindShaderRead(descriptorSet, 3);
        constBuffer.bind(descriptorSet, 4);
      }

      WorkGroupSize workGroupSize{8, 8, 1};
      if (hasT) {
        auto& computeUnit = context().computeUnitFactory().get(
            GLSL_SPV(addmm), descriptorSetLayout, workGroupSize);
        computeUnit.createCommandBuffer(descriptorSet);
        auto commandBuffer = computeUnit.commandBuffer();
        output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
        m1.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
        m2.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
        (*t).image()->addImageMemoryBarrierToShaderRead(commandBuffer);
        computeUnit.dispatchCommandBuffer(OW, OH, C_4, workGroupSize);
        computeUnit.endCommandBuffer();
        computeUnit.submitAndWaitCommandBuffer();
      } else {
        auto& computeUnit = context().computeUnitFactory().get(
            GLSL_SPV(mm), descriptorSetLayout, workGroupSize);
        computeUnit.createCommandBuffer(descriptorSet);
        auto commandBuffer = computeUnit.commandBuffer();
        output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
        m1.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
        m2.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
        computeUnit.dispatchCommandBuffer(OW, OH, C_4, workGroupSize);
        computeUnit.endCommandBuffer();
        computeUnit.submitAndWaitCommandBuffer();
      }
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}


pub fn mean(
        output: &mut VulkanTensor,
        input:  &VulkanTensor)  {
    
    todo!();
        /*
            auto isizes = input.sizes();
      i32 N = safe_downcast<i32>(isizes[0]);
      i32 C = safe_downcast<i32>(isizes[1]);
      i32 H = safe_downcast<i32>(isizes[2]);
      i32 W = safe_downcast<i32>(isizes[3]);

      auto device = context().device();
      struct ConstBlock {
        i32 W;
        i32 H;
      };
      ConstBlock cb{W, H};
      VBuffer constBuffer = makeUniformConstBuffer((void*)&cb, sizeof(cb));

      VkDescriptorSetLayout descriptorSetLayout{};
      VkDescriptorPool descriptorPool{};
      VkDescriptorSet descriptorSet{};
      vector<VkDescriptorType> descriptorTypes{
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
      createDescriptorSetLayoutSinglePool(
          device,
          descriptorTypes,
          &descriptorSetLayout,
          &descriptorPool,
          &descriptorSet);

      output.image()->bindStorageImage(descriptorSet, 0);
      input.image()->bindShaderRead(descriptorSet, 1);
      constBuffer.bind(descriptorSet, 2);

      WorkGroupSize workGroupSize{1, 1, 1};
      auto& computeUnit = context().computeUnitFactory().get(
          GLSL_SPV(mean2d), descriptorSetLayout, workGroupSize);
      computeUnit.createCommandBuffer(descriptorSet);
      auto commandBuffer = computeUnit.commandBuffer();
      output.image()->addImageMemoryBarrierToGeneral(commandBuffer);
      input.image()->addImageMemoryBarrierToShaderRead(commandBuffer);
      computeUnit.dispatchCommandBuffer(C, N, 1, workGroupSize);
      computeUnit.endCommandBuffer();
      computeUnit.submitAndWaitCommandBuffer();
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        */
}
