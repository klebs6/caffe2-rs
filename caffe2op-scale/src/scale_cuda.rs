crate::ix!();

impl ScaleOp<CUDAContext> {

    #[inline] pub fn run_on_cuda_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<at::Half, float>>::call(this, Input(0));
        */
    }
}

register_cuda_operator!{
    Scale, 
    ScaleOp<CUDAContext>
}
