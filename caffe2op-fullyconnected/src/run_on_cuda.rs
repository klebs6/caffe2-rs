crate::ix!();

#[inline] pub fn run_fully_connected_op_on_cudadevice<FullyConnectedOp>(
    float16_compute: bool,
    op: *mut FullyConnectedOp) -> bool 
{
    todo!();
    /*
        if (op->Input(0).template IsType<float>()) {
        return op->template DoRunWithType<
            float, // X
            float, // W
            float, // B
            float, // Y
            float>(); // Math
      } else if (op->Input(0).template IsType<at::Half>()) {
        if (float16_compute) {
          const cudaDeviceProp& prop = GetDeviceProperty(0);
          if (prop.major >= kFp16CUDADevicePropMajor) {
            return op->template DoRunWithType<
                at::Half, // X
                at::Half, // W
                at::Half, // B
                at::Half, // Y
                at::Half>(); // Math
          } else {
            LOG(INFO) << "CUDA Device does not support FP16 computation, "
                         "falling back to FP32.";
            return op->template DoRunWithType<
                at::Half, // X
                at::Half, // W
                at::Half, // B
                at::Half, // Y
                float>(); // Math
          }
        } else {
          return op->template DoRunWithType<
              at::Half, // X
              at::Half, // W
              at::Half, // B
              at::Half, // Y
              float>(); // Math
        }
      } else {
        CAFFE_THROW("Unsupported type");
      }
      return false;
    */
}

#[inline] pub fn run_fully_connected_gradient_op_on_cudadevice<FullyConnectedGradientOp>(
    float16_compute: bool,
    op: *mut FullyConnectedGradientOp) -> bool 
{
    todo!();
    /*
        if (op->Input(0).template IsType<float>()) {
        return op->template DoRunWithType<
            float, //  X
            float, //  W
            float, // dY
            float, //  B
            float, // dX
            float, // dW
            float, // dB
            float>(); // Math
      } else if (op->Input(0).template IsType<at::Half>()) {
        if (float16_compute) {
          const cudaDeviceProp& prop = GetDeviceProperty(0);
          if (prop.major >= kFp16CUDADevicePropMajor) {
            return op->template DoRunWithType<
                at::Half, //  X
                at::Half, //  W
                at::Half, // dY
                at::Half, //  B
                at::Half, // dX
                at::Half, // dW
                at::Half, // dB
                at::Half>(); // Math
          } else {
            LOG(INFO) << "CUDA Device does not support FP16 computation, "
                         "falling back to FP32.";
            return op->template DoRunWithType<
                at::Half, //  X
                at::Half, //  W
                at::Half, // dY
                at::Half, //  B
                at::Half, // dX
                at::Half, // dW
                at::Half, // dB
                float>(); // Math
          }
        } else {
          return op->template DoRunWithType<
              at::Half, //  X
              at::Half, //  W
              at::Half, // dY
              at::Half, //  B
              at::Half, // dX
              at::Half, // dW
              at::Half, // dB
              float>(); // Math
        }
      } else {
        CAFFE_THROW("Unsupported type");
      }
      return false;
    */
}
