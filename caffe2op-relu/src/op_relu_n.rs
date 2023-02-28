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

/**
  | ReluGradient takes both Y and dY and
  | uses this to update dX according to the
  | chain rule and derivatives of the rectified
  | linear function.
  |
  */
pub struct ReluNGradientFunctor<Context> {
    n:  f32,

    /**
      | Input: Y, dY
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{ReluNGradient, 2}

num_outputs!{ReluNGradient, 1}

args!{ReluNGradient, 
    0 => ("n", "the cap of forward op output")
}

allow_inplace!{ReluNGradient, vec![(1, 0)]}

impl<Context> ReluNGradientFunctor<Context> {

    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : n(op.GetSingleArgument<f32>("n", 6.0f)) 

        CAFFE_ENFORCE_GT(n, 0, "n should be greater than 0");
        */
    }
}

impl ReluNGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        y_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        y:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            const int size = std::accumulate(
          Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
      ConstEigenVectorArrayMap<T> Y_arr(Y, size);
      EigenVectorArrayMap<T>(dX, size) =
          (Y_arr > T(0) && Y_arr < T(n))
              .select(ConstEigenVectorArrayMap<T>(dY, size), T(0));
      return true;
        */
    }
}

#[inline] pub fn cost_inference_for_relun(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<2>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}

register_cpu_operator!{
    ReluN,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        ReluNFunctor<CPUContext>>}

register_cpu_operator!{
    ReluNGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        ReluNGradientFunctor<CPUContext>>}

pub struct GetReluNGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReluNGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{ReluN, GetReluNGradient}
