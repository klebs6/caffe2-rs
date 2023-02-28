crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
    GradientMakerBase,
    OperatorDef,
};

/**
  | MarginRankingCriterion takes two
  | input data X1 (Tensor),
  | 
  | X2 (Tensor), and label Y (Tensor) to
  | produce the loss (Tensor) where the
  | loss function, loss(X1, X2, Y) = max(0,
  | -Y * (X1 - X2) + margin), is applied to
  | the tensor elementwise.
  | 
  | If y == 1 then it assumed the first input
  | should be ranked higher (have a larger
  | value) than the second input, and vice-versa
  | for y == -1.
  |
  */
pub struct MarginRankingCriterionOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    margin: f32,
}

num_inputs!{MarginRankingCriterion, 3}

num_outputs!{MarginRankingCriterion, 1}

inputs!{MarginRankingCriterion, 
    0 => ("X1", "The left input vector as a 1-dim TensorCPU."),
    1 => ("X2", "The right input vector as a 1-dim TensorCPU."),
    2 => ("Y",  "The label as a 1-dim TensorCPU with int value of 1 or -1.")
}

outputs!{MarginRankingCriterion, 
    0 => ("loss", "The output loss with the same dimensionality as X1.")
}

args!{MarginRankingCriterion, 
    0 => ("margin", "The margin value as a float. Default is 1.0.")
}

impl<Context> MarginRankingCriterionOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "margin", margin_, 1.0)
        */
    }
}

impl MarginRankingCriterionOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X1 = Input(0);
      auto& X2 = Input(1);
      auto& Y = Input(2);

      CAFFE_ENFORCE_EQ(
          X1.numel(),
          X2.numel(),
          "The two inputs for computing ranking loss should have the same size.");
      CAFFE_ENFORCE_EQ(
          X1.numel(), Y.numel(), "The input and label should have the same size.");
      auto* loss = Output(0, X1.sizes(), at::dtype<float>());

      const float* X1data = X1.data<float>();
      const float* X2data = X2.data<float>();
      const int* Ydata = Y.data<int>();
      float* output = loss->template mutable_data<float>();
      for (int i = 0; i < X1.numel(); ++i) {
        output[i] = std::max(-Ydata[i] * (X1data[i] - X2data[i]) + margin_, 0.f);
      }
      return true;
        */
    }
}

/**
  | MarginRankingCriterionGradient
  | takes both X1, X2, Y and dY and uses them
  | to update dX1, and dX2 according to the
  | chain rule and derivatives of the loss
  | function.
  |
  */
pub struct MarginRankingCriterionGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    margin: f32,
}

num_inputs!{MarginRankingCriterionGradient, 4}

num_outputs!{MarginRankingCriterionGradient, 2}

impl<Context> MarginRankingCriterionGradientOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "margin", margin_, 1.0)
        */
    }
}

impl MarginRankingCriterionGradientOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X1 = Input(0);
      auto& X2 = Input(1);
      auto& Y = Input(2);
      auto& dLoss = Input(3);

      auto* dX1 = Output(0, X1.sizes(), at::dtype<float>());
      auto* dX2 = Output(1, X2.sizes(), at::dtype<float>());

      const float* X1data = X1.data<float>();
      const float* X2data = X2.data<float>();
      const int* Ydata = Y.data<int>();
      const float* dLoss_data = dLoss.data<float>();

      float* dX1_data = dX1->template mutable_data<float>();
      float* dX2_data = dX2->template mutable_data<float>();
      for (int i = 0; i < X1.numel(); ++i) {
        auto dist = -Ydata[i] * (X1data[i] - X2data[i]) + margin_;
        if (dist < 0.f) {
          dX1_data[i] = dX2_data[i] = 0.f;
        } else {
          dX1_data[i] = -Ydata[i] * dLoss_data[i];
          dX2_data[i] = Ydata[i] * dLoss_data[i];
        }
      }
      return true;
        */
    }
}

register_cpu_operator!{
    MarginRankingCriterion,
    MarginRankingCriterionOp<CPUContext>
}

register_cpu_operator!{
    MarginRankingCriterionGradient,
    MarginRankingCriterionGradientOp<CPUContext>
}

pub struct GetMarginRankingCriterionGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetMarginRankingCriterionGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "MarginRankingCriterionGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{MarginRankingCriterion, GetMarginRankingCriterionGradient}
