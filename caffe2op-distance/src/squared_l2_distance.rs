crate::ix!();

/**
  | Given two input float tensors X, Y, and
  | produces one output float tensor of
  | the L2 difference between X and Y that
  | is computed as ||(X - Y)^2 / 2||.
  |
  */
pub struct SquaredL2DistanceOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X, Y;
      | 
      | Output: Distance
      |
      */
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    SquaredL2Distance,
    SquaredL2DistanceOp<f32, CPUContext>
}

num_inputs!{SquaredL2Distance, 2}

num_outputs!{SquaredL2Distance, 1}

inputs!{SquaredL2Distance, 
    0 => ("X", "1D or 2D input tensor"),
    1 => ("Y", "1D or 2D input tensor (must have the same shape as X)")
}

outputs!{SquaredL2Distance, 
    0 => ("Z", "1D output tensor")
}

identical_type_and_shape_of_input_dim!{SquaredL2Distance, (0, 0)}

impl<T, Context> SquaredL2DistanceOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

