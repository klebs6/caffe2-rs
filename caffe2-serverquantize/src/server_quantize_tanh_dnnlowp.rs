crate::ix!();

pub struct TanhFunctor<T> {
    tanh:  Tanh<T>,
}

impl<T> TanhFunctor<T> {

    pub fn new() -> Self {
    
        todo!();
        /*
            : tanh_()
        */
    }

    #[inline] pub fn invoke(&mut self, 
        n: i32,
        x: *const T,
        y: *mut T)  {
        
        todo!();
        /*
            for (int i = 0; i < n; ++i) {
          y[i] = tanh_.Compute(x[i]);
        }
        */
    }
    
    #[inline] pub fn get_output_quantization_params(&self) -> TensorQuantizationParams {
        
        todo!();
        /*
            return tanh_.GetOutputQuantizationParams();
        */
    }
}

register_cpu_operator_with_engine!{
    Tanh,
    DNNLOWP,
    UnaryElementwiseWithArgsDNNLowPOp<u8, TanhFunctor<u8>>
}

register_cpu_operator_with_engine!{
    Int8Tanh,
    DNNLOWP,
    UnaryElementwiseWithArgsDNNLowPOp<u8, TanhFunctor<u8>>
}
