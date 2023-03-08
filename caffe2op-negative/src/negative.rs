crate::ix!();

/**
  | Computes the element-wise negative
  | of the input.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/negative_op.cc
  |
  */
pub struct NegativeFunctor<Context> { 

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

impl<Context> NegativeFunctor<Context> {
    
    #[inline] pub fn invoke<T>(
        &mut self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Neg(N, X, Y, context);
        return true;
        */
    }
}
