crate::ix!();

/**
  | Performs element-wise reciprocal
  | ($\1/x$) of input tensor $X$.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reciprocal_op.cc
  |
  */
pub struct ReciprocalFunctor<Context> {
    
    // Input: X, output: Y
    phantom: PhantomData<Context>,
}

impl<Context> ReciprocalFunctor<Context> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Inv(N, X, Y, context);
        return true;
        */
    }
}
