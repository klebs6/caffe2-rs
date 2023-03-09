crate::ix!();

pub struct SinFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

impl<Context> SinFunctor<Context> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        math::Sin(N, X, Y, context);
        return true;
        */
    }
}
