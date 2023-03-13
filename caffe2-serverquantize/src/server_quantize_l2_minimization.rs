crate::ix!();


/**
  | A quantization scheme that minimizes
  | L2 norm of quantization error.
  |
  */
pub struct L2ErrorMinimization {
    base: NormMinimization,
}

impl Default for L2ErrorMinimization {
    
    fn default() -> Self {
        todo!();
        /*
            : NormMinimization(L2
        */
    }
}

#[inline] pub fn l2minimization_kernelavx2(
    precision:     i32,
    bins:          *mut f32,
    nbins:         i32,
    bin_width:     f32,
    dst_bin_width: f32,
    start_bin:     i32) -> f32 {
    
    todo!();
    /*
    
    */
}
