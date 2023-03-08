crate::ix!();

#[inline] pub fn repeat_copy<T, Context>(
    repeat_n: usize,
    n:        usize,
    src:      *const T,
    dst:      *mut T,
    context:  *mut Context) 
{
    todo!();
    /*
        for (int i = 0; i < repeat_n; ++i) {
        context->template CopySameDevice<T>(n, src, dst + i * n);
      }
    */
}
