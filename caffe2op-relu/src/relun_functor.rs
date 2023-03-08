crate::ix!();

/**
  | Relu takes one input data (Tensor) and
  | produces one output data (Tensor) where
  | the rectified linear function, y = min(max(0,
  | x), n), is applied to the tensor elementwise.
  |
  */
pub struct ReluNFunctor<Context> {
    n:  f32,

    /**
      | Input: X
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{ReluN, 1}

num_outputs!{ReluN, 1}

inputs!{ReluN, 
    0 => ("X", "1D input tensor")
}

outputs!{ReluN, 
    0 => ("Y", "1D input tensor")
}

args!{ReluN, 
    0 => ("n", "the cap of output")
}

identical_type_and_shape!{ReluN}

cost_inference_function!{ReluN, CostInferenceForReluN }

allow_inplace!{ReluN, vec![(0, 0)]}

impl<Context> ReluNFunctor<Context> {

    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : n(op.GetSingleArgument<f32>("n", 6.0f)) 

        CAFFE_ENFORCE_GT(n, 0, "n should be greater than 0");
        */
    }
}

impl ReluNFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            EigenVectorMap<T>(Y, N) =
          ConstEigenVectorMap<T>(X, N).cwiseMax(T(0)).cwiseMin(T(n));
      return true;
        */
    }
}
