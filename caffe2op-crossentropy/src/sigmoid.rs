crate::ix!();

#[inline] pub fn sigmoid_xent_forward(lgt: f32, tgt: f32) -> f32 {
    
    todo!();
    /*
        return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
    */
}

#[inline] pub fn sigmoid_xent_backward(lgt: f32, tgt: f32) -> f32 {
    
    todo!();
    /*
        return tgt - 1. / (1. + exp(-lgt));
    */
}

#[inline] pub fn sigmoid_partition(lgt: f32) -> f32 {
    
    todo!();
    /*
        // computes log(1 + exp(lgt)) with only exp(x) function when x >= 0
      return lgt * (lgt >= 0) + log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
    */
}

#[inline] pub fn sigmoid_xent_forward_with_log_d_trick(lgt: f32, tgt: f32) -> f32 {
    
    todo!();
    /*
        return (2 * tgt - 1.) * (lgt - sigmoid_partition(lgt));
    */
}

#[inline] pub fn sigmoid_xent_backward_with_log_d_trick(lgt: f32, tgt: f32) -> f32 {
    
    todo!();
    /*
        return (2 * tgt - 1.) / (1. + exp(lgt));
    */
}

#[inline] pub fn unjoined_sigmoid_xent_forward(lgt: f32, tgt: f32) -> f32 {
    
    todo!();
    /*
        return lgt * tgt + (tgt - 1) * lgt * (lgt >= 0) -
          (1 - tgt) * log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
    */
}

#[inline] pub fn unjoined_sigmoid_xent_backward(lgt: f32, tgt: f32) -> f32 {
    
    todo!();
    /*
        return tgt - (1. - tgt) / (1. + exp(-lgt));
    */
}
