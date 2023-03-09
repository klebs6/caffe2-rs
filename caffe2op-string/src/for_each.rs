crate::ix!();

/**
  | ForEach is a unary functor that forwards
  | each element of the input array into
  | the elementwise Functor provided,
  | and gathers the results of each call
  | into the resulting array.
  | 
  | Use it as an adaptor if you want to create
  | a UnaryElementwiseOp that acts on each
  | element of the tensor per function call
  | -- this is reasonable for complex types
  | where vectorization wouldn't be much
  | of a gain, performance-wise.
  |
  */
pub struct ForEach<Functor> {
    functor:  Functor,
}

impl<Functor> ForEach<Functor> {

    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : functor(op)
        */
    }
    
    #[inline] pub fn invoke<In, Out, Context>(&mut self, 
        n:       i32,
        input:   *const In,
        out:     *mut Out,
        context: *mut Context) -> bool {
    
        todo!();
        /*
            for (int i = 0; i < n; ++i) {
          out[i] = functor(in[i]);
        }
        return true;
        */
    }
}
