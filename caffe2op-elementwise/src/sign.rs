crate::ix!();

/**
  | Computes sign for each element of the
  | input: -1, 0 or 1.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc
  |
  */
pub struct SignFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Sign, 1}

num_outputs!{Sign, 1}

inputs!{Sign, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor.")
}

outputs!{Sign, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

identical_type_and_shape_of_input!{Sign, 0}

should_not_do_gradient!{Sign}

impl<Context> SignFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &mut self,
        n: i32, 
        x: *const T, 
        y: *mut T, 
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Sign(N, X, Y, context);
            return true;
        */
    }
}
