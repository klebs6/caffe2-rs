crate::ix!();

#[inline] pub fn lSTMUnit<T, Context>(
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
    forget_bias: f32,
    context:     *mut Context)
{
    todo!();
    /*
        LstmUnitCpu<T>(
          N, D, t, H_prev, C_prev, X, seqLengths, drop_states, C, H, forget_bias);
    */
}
