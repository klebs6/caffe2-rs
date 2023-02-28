crate::ix!();

/**
  | Explicit initialize for float and AVX2
  | vectorization
  |
  */
#[inline] pub fn lstm_unit_implf32(
    n:           i32,
    d:           i32,
    t:           i32,
    h_prev:      *const f32,
    c_prev:      *const f32,
    x:           *const f32,
    seq_lengths: *const i32,
    drop_states: bool,
    c:           *mut f32,
    h:           *mut f32,
    forget_bias: f32)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn lstm_unit_gradient_implf32(
    n:           i32,
    d:           i32,
    t:           i32,
    c_prev:      *const f32,
    x:           *const f32,
    seq_lengths: *const i32,
    c:           *const f32,
    h:           *const f32,
    c_diff:      *const f32,
    h_diff:      *const f32,
    drop_states: bool,
    h_prev_diff: *mut f32,
    c_prev_diff: *mut f32,
    x_diff:      *mut f32,
    forget_bias: f32)  {
    
    todo!();
    /*
    
    */
}

/**
  | Define templated implementation fo
  | LSTM kernels on CPU supporting AVX2
  |
  */
#[inline] pub fn lstm_unit_impl_avx2_fma<T>(
    n:           i32,
    d:           i32,
    t:           i32,
    h_prev:      *const T,
    c_prev:      *const T,
    x:           *const T,
    seq_lengths: *const i32,
    drop_states: bool,
    c:           *mut T,
    h:           *mut T,
    forget_bias: f32)  {

    todo!();
    /*
        LstmUnitImpl(
          N, D, t, H_prev, C_prev, X, seqLengths, drop_states, C, H, forget_bias);
    */
}


#[inline] pub fn lstm_unit_gradient_impl_avx2_fma<T>(
    n:           i32,
    d:           i32,
    t:           i32,
    c_prev:      *const T,
    x:           *const T,
    seq_lengths: *const i32,
    c:           *const T,
    h:           *const T,
    c_diff:      *const T,
    h_diff:      *const T,
    drop_states: bool,
    h_prev_diff: *mut T,
    c_prev_diff: *mut T,
    x_diff:      *mut T,
    forget_bias: f32)  {

    todo!();
    /*
        LstmUnitGradientImpl(
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

/// Explicit initialize for float
#[inline] pub fn lstm_unit_impl_avx2_fmaf32(
    n:           i32,
    d:           i32,
    t:           i32,
    h_prev:      *const f32,
    c_prev:      *const f32,
    x:           *const f32,
    seq_lengths: *const i32,
    drop_states: bool,
    c:           *mut f32,
    h:           *mut f32,
    forget_bias: f32)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn lstm_unit_gradient_impl_avx2_fmaf32(
    n:           i32,
    d:           i32,
    t:           i32,
    c_prev:      *const f32,
    x:           *const f32,
    seq_lengths: *const i32,
    c:           *const f32,
    h:           *const f32,
    c_diff:      *const f32,
    h_diff:      *const f32,
    drop_states: bool,
    h_prev_diff: *mut f32,
    c_prev_diff: *mut f32,
    x_diff:      *mut f32,
    forget_bias: f32)  {
    
    todo!();
    /*
    
    */
}
