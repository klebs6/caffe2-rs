crate::ix!();

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
const TransposeWeight:     bool = true;

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
