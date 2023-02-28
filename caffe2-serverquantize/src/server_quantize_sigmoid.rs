crate::ix!();

use crate::{
    DEFAULT_MAX_ABS_ERR,
    TensorQuantizationParams,
    Tanh
};

/**
  | sigmoid(x) = (tanh(x/2) + 1)/2
  | 
  | Quantized sigmoid is computed as tanh
  | under the hood, we just use different
  | input/output quantization parameters.
  | u8, u16, i32
  |
  */
pub struct Sigmoid<T> {
    
    /// = Tanh<T>::DEFAULT_NUM_IN_BITS;
    num_in_bits:   i32,

    /// = Tanh<T>::DEFAULT_NUM_OUT_BITS;
    num_out_bits:  i32,

    tanh:          Tanh<T>,
    in_qparams:    TensorQuantizationParams,
    out_qparams:   TensorQuantizationParams,
}

impl<T> Sigmoid<T> {

    #[inline] pub fn get_input_quantization_params(&self) -> TensorQuantizationParams {
        
        todo!();
        /*
            return in_qparams_;
        */
    }
    
    #[inline] pub fn get_output_quantization_params(&self) -> TensorQuantizationParams {
        
        todo!();
        /*
            return out_qparams_;
        */
    }
    
    pub fn new(max_abs_err: Option<f64>) -> Self {

        let max_abs_err = max_abs_err.unwrap_or(DEFAULT_MAX_ABS_ERR);
    
        todo!();
        /*
            : tanh_(max_abs_err) 

                float x_sq = tanh_.GetSaturationRegionBegin();

                in_qparams_.scale = 2 * x_sq / ((1 << (num_in_bits_ - 1)) - 1);
                in_qparams_.zero_point = 1 << (num_in_bits_ - 1);
                in_qparams_.precision = num_in_bits_;
                // -2 x_sq is mapped to -127, 0 is mapped to 0, 2 x_sq is mapped to 127

                out_qparams_.scale = 0.5 / ((1 << (num_out_bits_ - 1)) - 1);
                out_qparams_.zero_point = 0;
                out_qparams_.precision = num_out_bits_;
                // 0 is mapped to 0, 1/2 is mapped to 127, 1 is mapped to 254
        */
    }
    
    #[inline] pub fn compute(&self, x: T) -> T {
    
        todo!();
        /*
            T temp = tanh_.Compute(x);
                assert(temp >= 1);
                assert(temp < (1 << num_out_bits_));
                return temp - 1;
        */
    }
}
