crate::ix!();

pub struct SigmoidFunctor<T> {
    sigmoid:  Sigmoid<T>,
}

impl<T> SigmoidFunctor<T> {

    pub fn new() -> Self {
    
        todo!();
        /*
            : sigmoid_()
        */
    }
    
    #[inline] pub fn invoke(&mut self, 
        n: i32,
        x: *const T,
        y: *mut T)  {

        todo!();
        /*
            for (int i = 0; i < n; ++i) {
          y[i] = sigmoid_.Compute(x[i]);
        }
        */
    }
    
    #[inline] pub fn get_output_quantization_params(&self) -> TensorQuantizationParams {
        
        todo!();
        /*
            return sigmoid_.GetOutputQuantizationParams();
        */
    }
}

register_cpu_operator_with_engine!{
    Sigmoid,
    DNNLOWP,
    UnaryElementwiseWithArgsDNNLowPOp<
        u8,
        SigmoidFunctor<u8>>
}

register_cpu_operator_with_engine!{
    Int8Sigmoid,
    DNNLOWP,
    UnaryElementwiseWithArgsDNNLowPOp<
        u8,
        SigmoidFunctor<u8>>
}
