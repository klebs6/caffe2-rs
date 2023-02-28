crate::ix!();

#[test] fn elu_functor_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Elu",
        ["X"],
        ["Y"],
        alpha=1.1
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[ 0.35339102  1.1860217  -0.10710736]
     [-3.1173866  -0.1889988  -0.20330353]
     [ 1.8525308  -0.368949    0.506277  ]]

    Y:
     [[ 0.35339102  1.1860217  -0.11172786]
     [-1.0513     -0.18943374 -0.20236646]
     [ 1.8525308  -0.33939326  0.506277  ]]
    */
}

/**
 | This op implements the exponential linear unit
 | (ELU) activation function as described in [Fast
 | and Accurate 
 |
 | Deep Network Learning by Exponential Linear Units
 | (ELUs)]
 |
 | (https://arxiv.org/abs/1511.07289). 
 |
 | The op takes an input tensor $X$ of arbitrary
 | shape, computes the elementwise elu operation, and
 | returns a vector $Y$ of the same shape as output. 
 |
 | The alpha parameter may be passed as an argument,
 | but defaults to 1. 
 |
 | The elu operation is defined as
 |
 | $$y=f(x) =\begin{cases}\alpha(e^x-1) & x < 0 \\
 | x & otherwise\end{cases}$$
 |
 | Github Links:
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.h
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.cc
 */
pub struct EluFunctor<Context> {
    alpha: f32,

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

impl<Context> EluFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 1.0f))
        */
    }
}

/**
  | EluGradient takes both Y and dY and uses
  | this to update dX according to the chain
  | rule and derivatives of the rectified
  | linear function.
  |
  */
pub struct EluGradientFunctor<Context> {
    alpha: f32,

    /**
      | Input: Y, dY,
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{EluGradient, 2}

num_outputs!{EluGradient, 1}

allow_inplace!{EluGradient, vec![(1, 0)]}

impl<Context> EluGradientFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 1.0f))
        */
    }
}

impl<CPUContext> EluFunctor<CPUContext> {
    #[inline] pub fn invoke<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> X_arr(X, N);
          EigenVectorMap<T>(Y, N) =
              (X_arr < 0).select(alpha * (X_arr.exp() - T(1)), X_arr);
          return true;
        */
    }
}

impl<CPUContext> EluGradientFunctor<CPUContext> {
    #[inline] pub fn forward<T>(
        &self,
        y_dims:    &Vec<i32>,
        dY_dims:   &Vec<i32>,
        y:         *const T,
        dY:        *const T,
        dX:        *mut T,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> Y_arr(Y, size);
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          EigenVectorArrayMap<T>(dX, size) =
              (Y_arr < 0).select(dY_arr * (Y_arr + alpha), dY_arr);
          return true;
        */
    }
}

register_cpu_operator!{
    Elu,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        EluFunctor<CPUContext>>}

register_cpu_gradient_operator!{
    EluGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        EluGradientFunctor<CPUContext>>}

num_inputs!{Elu, 1}

num_outputs!{Elu, 1}

inputs!{Elu, 
    0 => ("X", "1D input tensor of data to be operated on.")
}

outputs!{Elu, 
    0 => ("Y", "1D input tensor, calculated as described above.")
}

args!{Elu, 
    0 => ("alpha", "*(type: float; default: 1.0)* Defines alpha parameter used in calculation.")
}

allow_inplace!{Elu, vec![(0, 0)]}

identical_type_and_shape!{Elu}

inherit_onnx_schema!{Elu}

pub struct GetEluGradient {}

impl GetGradientDefs for GetEluGradient {

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

register_gradient!{Elu, GetEluGradient}
