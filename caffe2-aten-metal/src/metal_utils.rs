crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/MetalUtils.h]

#[cfg(any(__ARM_NEON__,__ARM_NEON))]      pub type fp16_t = float16_t;
#[cfg(not(any(__ARM_NEON__,__ARM_NEON)))] pub type fp16_t = u16;

pub fn fp_32to_fp16(src: &Vec<f32>) -> Vec<f16> {
    
    todo!();
        /*
        
        */
}

pub fn fp_16to_fp32(src: &Vec<f16>) -> Vec<f32> {
    
    todo!();
        /*
        
        */
}

pub fn nchw_tonc4(
        src:   *const f32,
        sizes: &Vec<i64>) -> Vec<f32> {
    
    todo!();
        /*
        
        */
}

pub fn nc4tonchw(
        src:   *const f32,
        sizes: &Vec<i64>) -> Vec<f32> {
    
    todo!();
        /*
        
        */
}

/**
  | When copying the result back to a CPU tensor,
  | the memory format becomes NCHW.
  |
  | Thus,we compute the strides based on contiguous
  | memory format.
  |
  */
#[inline] pub fn compute_strides(sizes: &Vec<i64>) -> Vec<i64> {
    
    todo!();
        /*
            const auto dim = sizes.size();
      vector<i64> strides(dim, 0);
      if (dim > 0) {
        const auto last_idx = dim - 1;
        strides[last_idx] = 1;
        for (int i = last_idx - 1; i >= 0; --i) {
          strides[i] = strides[i + 1] * max<i64>(sizes[i + 1], 1);
        }
      }
      return strides;
        */
}

#[inline] pub fn get_tensor_impl_storage(tensor: &Tensor) -> &mut MetalTensorImplStorage {
    
    todo!();
        /*
            using MetalTensorImpl = MetalTensorImpl<MetalTensorImplStorage>;
      TORCH_CHECK(tensor.is_metal());
      MetalTensorImpl* impl =
          static_cast<MetalTensorImpl*>(tensor.unsafeGetTensorImpl());
      return impl->unsafe_opaque_handle();
        */
}

#[inline] pub fn make_tensor(
        mt:      MetalTensorImplStorage,
        options: &TensorOptions) -> Tensor {
    
    todo!();
        /*
            using MetalTensorImpl = MetalTensorImpl<MetalTensorImplStorage>;
      auto sizes = mt.sizes(); // sizes is stored in TensorImpl
      auto strides = mt.strides(); // strides is stored in MetalTensorImpl
      return make_tensor<MetalTensorImpl>(
          DispatchKeySet(DispatchKey::Metal),
          options.dtype(),
          Device(kMetal),
          move(mt),
          vector<i64>(sizes.begin(), sizes.end()),
          vector<i64>(strides.begin(), strides.end()));
        */
}

#[inline] pub fn get_command_buffer_from_tensor(tensor: &Tensor) -> *mut MetalCommandBuffer {
    
    todo!();
        /*
            TORCH_CHECK(tensor.is_metal());
      auto implStorage = getTensorImplStorage(tensor);
      MetalCommandBuffer* cmdBuffer = implStorage.texture()->commandBuffer();
      if (!cmdBuffer || !cmdBuffer.valid) {
        cmdBuffer = [MetalCommandBuffer currentBuffer];
      }
      return cmdBuffer;
        */
}

pub fn make_mtl_buffer<T>(src: &Vec<T>) -> id<MTLBuffer> {

    todo!();
        /*
            id<MTLBuffer> buffer = [[MPSCNNContext sharedInstance].device
              newBufferWithLength:src.size() * sizeof(T)
                          options:MTLResourceOptionCPUCacheModeWriteCombined];
        memcpy(buffer.contents, src.data(), src.size() * sizeof(T));
        return buffer;
        */
}

#[inline] pub fn make_mtl_buffer_bytes(bytes: i64) -> id<MTLBuffer> {
    
    todo!();
        /*
            id<MTLBuffer> buffer = [[MPSCNNContext sharedInstance].device
              newBufferWithLength:bytes
                          options:MTLResourceOptionCPUCacheModeWriteCombined];
        return buffer;
        */
}
