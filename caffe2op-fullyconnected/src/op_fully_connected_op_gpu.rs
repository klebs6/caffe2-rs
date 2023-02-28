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

pub const NoTransposeWeight: bool = false;

impl FullyConnectedOp<CUDAContext, DefaultEngine, NoTransposeWeight> {

    /**
      | The RunFullyConnectedOpOnCUDADevice
      | Function will use the pointer of current
      | op and the DoRunWithType will make sure to
      | run the correct things.
      */
    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return RunFullyConnectedOpOnCUDADevice(float16_compute_, this);
        */
    }
}

impl FullyConnectedGradientOp<CUDAContext, DefaultEngine, TransposeWeight> {

    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return RunFullyConnectedGradientOpOnCUDADevice(float16_compute_, this);
        */
    }
}
    
impl FullyConnectedGradientOp<CUDAContext, DefaultEngine, DontTransposeWeight> {
    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return RunFullyConnectedGradientOpOnCUDADevice(float16_compute_, this);
        */
    }
}

const DontTransposeWeight: bool = false;
const TransposeWeight: bool = true;

/**
  | Require these to be defined otherwise
  | TensorCore FC ops will end up calling the
  | default FC implementation which doesn't have
  | fp16 support...
  */
impl FullyConnectedOp<CUDAContext, TensorCoreEngine, TransposeWeight> {

    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return RunFullyConnectedOpOnCUDADevice(false /* float16_compute */, this);
        */
    }
}
    
impl FullyConnectedOp<CUDAContext, TensorCoreEngine, DontTransposeWeight> {
    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return RunFullyConnectedOpOnCUDADevice(false /* float16_compute */, this);
        */
    }
}

impl FullyConnectedGradientOp<CUDAContext, TensorCoreEngine, TransposeWeight> {

    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return RunFullyConnectedGradientOpOnCUDADevice(
          false /* float16_compute */, this);
        */
    }
}

impl FullyConnectedGradientOp<CUDAContext, TensorCoreEngine, DontTransposeWeight> {

    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return RunFullyConnectedGradientOpOnCUDADevice(
          false /* float16_compute */, this);
        */
    }
}

register_cuda_operator!{FC, FullyConnectedOp<CUDAContext>}

register_cuda_operator!{FCGradient, FullyConnectedGradientOp<CUDAContext>}

register_cuda_operator!{
    FCTransposed,
    FullyConnectedOp<
        CUDAContext,
        DefaultEngine,
        DontTransposeWeight>
}

register_cuda_operator!{
    FCTransposedGradient,
    FullyConnectedGradientOp<
        CUDAContext,
        DefaultEngine,
        DontTransposeWeight>
}

register_cuda_operator_with_engine!{
    FC,
    TENSORCORE,
    FullyConnectedOp<CUDAContext, TensorCoreEngine>
}

register_cuda_operator_with_engine!{
    FCGradient,
    TENSORCORE,
    FullyConnectedGradientOp<CUDAContext, TensorCoreEngine>
}

register_cuda_operator_with_engine!{
    FCTransposed,
    TENSORCORE,
    FullyConnectedOp<
        CUDAContext,
        TensorCoreEngine,
        DontTransposeWeight>}

register_cuda_operator_with_engine!{
    FCTransposedGradient,
    TENSORCORE,
    FullyConnectedGradientOp<
        CUDAContext,
        TensorCoreEngine,
        DontTransposeWeight>}
