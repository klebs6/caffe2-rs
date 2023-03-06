crate::ix!();

#[inline] pub fn lstm_unit_gradient<T, Context>(
    n:             i32,
    d:             i32,
    t:             i32,
    c_prev:        *const T,
    x:             *const T,
    seq_lengths:   *const i32,
    c:             *const T,
    h:             *const T,
    c_diff:        *const T,
    h_diff:        *const T,
    drop_states:   bool,
    h_prev_diff:   *mut T,
    c_prev_diff:   *mut T,
    x_diff:        *mut T,
    forget_bias:   f32,
    context:       *mut Context) 
{
    todo!();
    /*
        LstmUnitGradientCpu<T>(
          N,
          D,
          t,
          C_prev,
          X,
          seqLengths,
          C,
          H,
          C_diff,
          H_diff,
          drop_states,
          H_prev_diff,
          C_prev_diff,
          X_diff,
          forget_bias);
    */
}
