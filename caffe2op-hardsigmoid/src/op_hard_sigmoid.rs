crate::ix!();

/**
  | Applies hard sigmoid operation to the
  | input data element-wise.
  | 
  | The HardSigmoid operation takes one
  | input $X$, produces one output $Y$,
  | and is defined as:
  | 
  | $$Y = max(0,min(1,x * alpha + beta))$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hard_sigmoid_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hard_sigmoid_op.cc
  |
  */
pub struct HardSigmoidFunctor<Context> {
    alpha:   f32,
    beta:    f32,

    /**
      | Input: X
      | 
      | Output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{HardSigmoid, 1}

num_outputs!{HardSigmoid, 1}

inputs!{HardSigmoid, 
    0 => ("X", "1D input tensor")
}

outputs!{HardSigmoid, 
    0 => ("Y", "1D output tensor with same shape as input")
}

args!{HardSigmoid, 
    0 => ("alpha", "float: the slope of the function. Defaults to 0.2"),
    1 => ("beta", "float: the bias value of the function. Defaults to 0.5")
}

identical_type_and_shape!{HardSigmoid}

allow_inplace!{HardSigmoid, vec![(0, 0)]}

cost_inference_function!{HardSigmoid, CostInferenceForHardSigmoid }

inherit_onnx_schema!{HardSigmoid}

impl<Context> HardSigmoidFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 0.2f)),
            beta(op.GetSingleArgument<float>("beta", 0.5f))
        */
    }
}

impl HardSigmoidFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(
        &self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool 
    {
        todo!();
        /*
            EigenVectorArrayMap<T>(Y, N) =
              (ConstEigenVectorArrayMap<T>(X, N) * T(alpha) + T(beta))
                  .cwiseMin(T(1))
                  .cwiseMax(T(0));
          return true;
        */
    }
}

/**
  | HardSigmoidGradient takes both Y and
  | dY as well as an argument alpha and uses
  | this to update dX according to the chain
  | rule and derivatives of the hard sigmoid
  | function.
  |
  */
pub struct HardSigmoidGradientFunctor<Context> {

    alpha: f32,

    /**
      | Input: Y, dY
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{HardSigmoidGradient, 2}

num_outputs!{HardSigmoidGradient, 1}

allow_inplace!{HardSigmoidGradient, vec![(1, 0)]}

impl<Context> HardSigmoidGradientFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 0.2f))
        */
    }
}


impl HardSigmoidGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        y_dims:   &Vec<i32>,
        dY_dims:  &Vec<i32>,
        y:        *const T,
        dY:       *const T,
        dX:       *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> Y_arr(Y, size);
          EigenVectorArrayMap<T>(dX, size) =
              (Y_arr > T(0) && Y_arr < T(1))
                  .select(ConstEigenVectorArrayMap<T>(dY, size) * alpha, T(0));
          return true;
        */
    }
}

#[inline] pub fn cost_inference_for_hard_sigmoid(
    def: &OperatorDef,
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<4>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}

register_cpu_operator!{
    HardSigmoid,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        HardSigmoidFunctor<CPUContext>>
}

register_cpu_operator!{
    HardSigmoidGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        HardSigmoidGradientFunctor<CPUContext>>
}

#[test] fn hard_sigmoid_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "HardSigmoid",
        ["X"],
        ["Y"],
        alpha = 0.2,
        beta = 0.5,
    )

    workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
    print("input:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("sigmoid:", workspace.FetchBlob("Y"))

    input: [ 1.5744036   0.31632107  1.7842269   1.4450722  -2.1726978 ]
    hard_sigmoid: [ 0.81488073,  0.56326419,  0.85684538,  0.78901446,  0.06546044]
    */
}

pub struct GetHardSigmoidGradient;

impl GetGradientDefs for GetHardSigmoidGradient {

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

register_gradient!{HardSigmoid, GetHardSigmoidGradient}
