crate::ix!();

impl ImageInputOp<CUDAContext> {

    #[inline] pub fn apply_transform_on_gpu(
        &mut self, 
        dims: &Vec<i64>, 
        ty:   &Device) -> bool 
    {
        todo!();
        /*
            // GPU transform kernel allows explicitly setting output type
          if (output_type_ == TensorProto_DataType_FLOAT) {
            auto* image_output =
                OperatorStorage::OutputTensor(0, dims, at::dtype<float>().device(type));
            TransformOnGPU<uint8_t, float, CUDAContext>(
                prefetched_image_on_device_,
                image_output,
                mean_gpu_,
                std_gpu_,
                &context_);
          } else if (output_type_ == TensorProto_DataType_FLOAT16) {
            auto* image_output =
                OperatorStorage::OutputTensor(0, dims, at::dtype<at::Half>().device(type));
            TransformOnGPU<uint8_t, at::Half, CUDAContext>(
                prefetched_image_on_device_,
                image_output,
                mean_gpu_,
                std_gpu_,
                &context_);
          } else {
            return false;
          }
          return true;
        */
    }
}
