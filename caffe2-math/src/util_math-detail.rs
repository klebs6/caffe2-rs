crate::ix!();

/**
  | proxy to a class because of partial specialization
  | limitations for functions
  |
  */
pub struct ScaleImpl<T,Context,const FixedSize: i32> { }

impl ScaleImpl<T,Context,const FixedSize: i32> {
    
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
                *y = *x * alpha;
                */
            }
            _ => {
                scale(n, alpha, x, y, context);
            }
        }
    }
}

pub struct AxpyImpl<T,Context,const FixedSize: i32> { }

impl AxpyImpl<T,Context,const FixedSize: i32> {
    
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
                axpy(n, alpha, x, y, context);
            }
        }
    }
}

#[inline] pub fn scale_fixed_size<T, Context, const FixedSize: i32>(
    n:       i32,
    alpha:   f32,
    x:       *const T,
    y:       *mut T,
    context: *mut Context)  {
    todo!();
    /*
        detail::ScaleImpl<T, Context, FixedSize>()(N, alpha, x, y, context);
    */
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

