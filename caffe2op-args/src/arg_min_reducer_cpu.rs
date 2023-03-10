crate::ix!();

impl ArgMinReducer<CPUContext> {

    #[inline] pub fn invoke<T>(
        &mut self,
        prev_size: i32,
        next_size: i32,
        n:         i32,
        x:         *const T,
        y:         *mut i64,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            ComputeArgImpl(prev_size, next_size, n, std::less<T>(), X, Y, context);
          return true;
        */
    }
}

register_cpu_operator!{
    ArgMin, 
    ArgOp<CPUContext, ArgMinReducer<CPUContext>>
}
