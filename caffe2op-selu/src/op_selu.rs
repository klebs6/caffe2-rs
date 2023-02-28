crate::ix!();

use crate::{
    OperatorStorage,
    GradientMakerBase,
    OperatorDef,
    CPUContext
};

#[test] fn selu_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Selu",
        ["X"],
        ["Y"],
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[ 1.1613879  -0.27111396 -1.2076733 ]
     [ 1.3442237  -1.0701777   1.2070968 ]
     [ 0.23810555  0.9740916  -1.7872391 ]]

    Y:
     [[ 1.2202715  -0.4174965  -1.2326177 ]
     [ 1.4123772  -1.1551634   1.2682979 ]
     [ 0.25017774  1.023479   -1.4637551 ]]
    */
}

/**
  | The *Selu* op takes one input tensor
  | $X$, an argument $alpha$, an argument
  | $scale$, and produces one output tensor
  | $Y$ of the same shape as $X.$ The op performs
  | the element wise *Selu* operation,
  | defined as
  | 
  | $$y=selu(x) =\begin{cases}scale
  | (\alpha e^{x} - \alpha) & x < 0\\scale
  | * x & otherwise\end{cases}$$
  | 
  | The default value of *alpha* is 1.6732632423543772848170429916717
  | and the default value of *scale* is 1.0507009873554804934193349852946.
  | See [Self-Normalizing Neural
  | 
  | Networks](https://arxiv.org/abs/1706.02515)
  | for more information.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.cc
  |
  */
pub struct SeluOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,

    alpha:   T,
    lambda:  T,

    /*
      | Input: X;
      | 
      | output: Y
      |
      */
}

num_inputs!{Selu, 1}

num_outputs!{Selu, 1}

inputs!{Selu, 
    0 => ("X", "Input tensor of data to be operated on.")
}

outputs!{Selu, 
    0 => ("Y", "Output tensor with same shape as input.")
}

args!{Selu, 
    0 => ("alpha", "*(type: float; default: 1.673263~)* Alpha constant in equation."),
    1 => ("scale", "*(type: float; default: 1.050700~; must be > 1.0)* Scale constant in equation.")
}

identical_type_and_shape!{Selu}

allow_inplace!{Selu, vec![(0, 0)]}

inherit_onnx_schema!{Selu}

impl<T,Context> SeluOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>(
            "alpha", 1.6732632423543772848170429916717f);
        lambda_ = this->template GetSingleArgument<T>(
            "scale", 1.0507009873554804934193349852946f);
        // In the paper "scale" is named "lambda", but "lambda" is a reserved
        // keyword in python
        CAFFE_ENFORCE_GT(lambda_, 1.0);
        */
    }
}

/**
  | SeluGradient takes both Y and dY and
  | uses this to update dX according to the
  | chain rule and derivatives of the selu
  | function.
  |
  */
pub struct SeluGradientOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    alpha:   T,
    lambda:  T,

    /*
      | Input: Y, dY;
      | 
      | output: dX
      |
      */
}

num_inputs!{SeluGradient, 2}

num_outputs!{SeluGradient, 1}

inputs!{SeluGradient, 
    0 => ("Y", "input tensor"),
    1 => ("dY", "input tensor")
}

args!{SeluGradient, 
    0 => ("alpha", "(float) default to 1.6732~; affects the activation function itself. This should go with the weight initialization in the paper.  See https://arxiv.org/abs/1706.02515 "),
    1 => ("scale", "(float) default to 1.0507~; affects the activation function itself.")
}

allow_inplace!{SeluGradient, vec![(1, 0)]}

impl<T,Context> SeluGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>(
            "alpha", 1.6732632423543772848170429916717f);
        lambda_ = this->template GetSingleArgument<T>(
            "scale", 1.0507009873554804934193349852946f);
        CAFFE_ENFORCE_GT(lambda_, 1.0);
        */
    }
}

impl SeluOp<f32, CPUContext> {

    #[inline] pub fn run_on_deviceA(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());

      ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.numel());
      EigenVectorArrayMap<float> Yvec(
          Y->template mutable_data<float>(), Y->numel());
      Yvec = lambda_ * (Xvec > 0).select(Xvec, (alpha_ * Xvec.exp() - alpha_));
      return true;
        */
    }
    
    #[inline] pub fn run_on_deviceB(&mut self) -> bool {
        
        todo!();
        /*
            auto& Y = Input(0);
      auto& dY = Input(1);

      CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
      auto* dX = Output(0, Y.sizes(), at::dtype<float>());

      ConstEigenVectorArrayMap<float> Yvec(Y.data<float>(), Y.numel());
      ConstEigenVectorArrayMap<float> dYvec(dY.data<float>(), dY.numel());
      EigenVectorArrayMap<float> dXvec(
          dX->template mutable_data<float>(), dX->numel());

      const float la = lambda_ * alpha_;
      dXvec = (Yvec > 0).select(lambda_ * dYvec, dYvec * (Yvec + la));
      return true;
        */
    }
}

register_cpu_operator!{Selu, SeluOp<float, CPUContext>}

register_cpu_operator!{SeluGradient, SeluGradientOp<float, CPUContext>}

///-------------------------------
pub struct GetSeluGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSeluGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            vector<string>{O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{Selu, GetSeluGradient}
