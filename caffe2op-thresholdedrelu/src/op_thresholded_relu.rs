crate::ix!();

use crate::{
    OperatorStorage,
    GradientMakerBase,
    CPUContext,
    OperatorDef,
};


/**
  | ThresholdedRelu takes one input data
  | (Tensor) and produces one output data
  | (Tensor) where the rectified linear
  | function, y = x for x > alpha, y = 0 otherwise,
  | is applied to the tensor elementwise.
  |
  */
pub struct ThresholdedReluOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    alpha:   T,

    /*
      | Input: X,
      | 
      | output: Y
      |
      */
}

register_cpu_operator!{ThresholdedRelu, ThresholdedReluOp<float, CPUContext>}

num_inputs!{ThresholdedRelu, 1}

num_outputs!{ThresholdedRelu, 1}

inputs!{ThresholdedRelu, 
    0 => ("X", "1D input tensor")
}

outputs!{ThresholdedRelu, 
    0 => ("Y", "1D input tensor")
}

args!{ThresholdedRelu, 
    0 => ("alpha", "(float) defaults to 1.0.")
}

identical_type_and_shape!{ThresholdedRelu}

cost_inference_function!{ThresholdedRelu, 
    PointwiseCostInference::<2>
}

allow_inplace!{ThresholdedRelu, vec![(0, 0)]}

impl<T,Context> ThresholdedReluOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>("alpha", 1.0);
        */
    }
}

impl ThresholdedReluOp<f32,CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());

      ConstEigenVectorArrayMap<float> Xvec(X.data<float>(), X.numel());
      EigenVectorArrayMap<float> Yvec(
          Y->template mutable_data<float>(), Y->numel());
      Yvec = (Xvec > alpha_).select(Xvec, 0.f);
      /* Naive implementation
      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      for (int i = 0; i < X.size(); ++i) {
        Xdata[i] -= alpha_;
        Ydata[i] = std::max(Xdata[i], 0.0f);
      }
      */
      return true;
        */
    }
}

/**
  | ThresholdedReluGradient takes both
  | Y and dY and uses this to update dX according
  | to the chain rule and derivatives of
  | the rectified linear function.
  |
  */
pub struct ThresholdedReluGradientOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    alpha:   T,

    /*
      | Input: Y, dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{ThresholdedReluGradient, 2}

num_outputs!{ThresholdedReluGradient, 1}

allow_inplace!{ThresholdedReluGradient, vec![(1, 0)]}

impl<T,Context> ThresholdedReluGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>("alpha", 1.0);
        */
    }
}

impl ThresholdedReluGradientOp<f32,CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& Y = Input(0);
      auto& dY = Input(1);

      CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
      auto* dX = Output(0, Y.sizes(), at::dtype<float>());

      const float* Ydata = Y.data<float>();
      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      EigenVectorArrayMap<float> dXvec(dXdata, dX->numel());
      ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.numel());
      ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.numel());
      dXvec = dYvec * Yvec.cwiseSign();
      /* Non vectorized implementation
      for (int i = 0; i < Y.size(); ++i) {
        dXdata[i] = Ydata[i] > 0 ? dYdata[i] : 0;
      }
      */
      return true;
        */
    }
}

register_cpu_operator!{
    ThresholdedReluGradient,
    ThresholdedReluGradientOp<f32, CPUContext>
}

pub struct GetThresholdedReluGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetThresholdedReluGradient<'a> {
    
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

register_gradient!{ThresholdedRelu, GetThresholdedReluGradient}
