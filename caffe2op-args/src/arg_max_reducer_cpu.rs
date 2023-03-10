crate::ix!();

impl ArgMaxReducer<CPUContext> {

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
            ComputeArgImpl(prev_size, next_size, n, std::greater<T>(), X, Y, context);
          return true;
        */
    }
}

register_cpu_operator!{
    ArgMax, 
    ArgOp<CPUContext, ArgMaxReducer<CPUContext>>
}
