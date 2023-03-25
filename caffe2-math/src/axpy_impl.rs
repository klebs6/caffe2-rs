crate::ix!();

pub struct AxpyImpl<T,Context,const FixedSize: i32> { 
    p_x: PhantomData<T>,
    p_y: PhantomData<Context>,
}

#[inline] pub fn axpy_fixed_size<T, Context, const FixedSize: i32>(
    n:       i32,
    alpha:   f32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
        detail::AxpyImpl<T, Context, FixedSize>()(N, alpha, x, y, context);
    */
}

impl<T,Context,const FixedSize: i32> AxpyImpl<T,Context,FixedSize> {
    
    #[inline] pub fn invoke(&mut self, 
        n:       i32,
        alpha:   f32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context)  
    {
        match FixedSize {
            1 => {
                todo!();
                /*
                   DCHECK_EQ(N, 1);
                 *y += *x * alpha;
                 */
            }
            _ => {
                axpy(n as i64, alpha, x, y, context);
            }
        }
    }
}

