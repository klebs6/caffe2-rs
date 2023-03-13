crate::ix!();

pub type ConvertFnType<T> = fn(
    dst: *mut f32, 
    src: *const T, 
    N: libc::size_t) -> ();


#[inline] pub fn convertfp_16fp32(dst: *mut f32, src: *const f16, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; i++) {
        dst[i] = src[i];
      }
    */
}

#[inline] pub fn convertfp_32fp16(dst: *mut f16, src: *const f32, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; i++) {
        dst[i] = src[i];
      }
    */
}
