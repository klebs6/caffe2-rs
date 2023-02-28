crate::ix!();

use crate::{
    CPUContext,
    OperatorDef,
    OperatorStorage,
    GradientMakerBase,
};

/**
 | CosineEmbeddingCriterion takes two inputs: the
 | similarity value and the label, and computes the
 | elementwise criterion output as
 |
 | output = 1 - s,               if y == 1
 | max(0, s - margin),  if y == -1
 */
pub struct CosineEmbeddingCriterionOp<Context> {
    storage: OperatorStorage,
    context: Context,
    margin:  f32,
}

num_inputs!{CosineEmbeddingCriterion, 2}

num_outputs!{CosineEmbeddingCriterion, 1}

inputs!{CosineEmbeddingCriterion, 
    0 => ("S", "The cosine similarity as a 1-dim TensorCPU."),
    1 => ("Y", "The label as a 1-dim TensorCPU with int value of 1 or -1.")
}

outputs!{CosineEmbeddingCriterion, 
    0 => ("loss", "The output loss with the same dimensionality as S.")
}

impl CosineEmbeddingCriterionOp<CPUContext> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
           : Operator<Context>(std::forward<Args>(args)...),
           OP_SINGLE_ARG(float, "margin", margin_, 0.0)
           */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& S = Input(0);
      auto& Y = Input(1);

      CAFFE_ENFORCE(
          S.numel() == Y.numel(),
          "The embedding and label should have the same size.");
      auto* output = Output(0, S.sizes(), at::dtype<float>());

      const float* Sdata = S.data<float>();
      const int* Ydata = Y.data<int>();
      float* output_data = output->template mutable_data<float>();
      for (int i = 0; i < S.numel(); ++i) {
        output_data[i] =
            Ydata[i] == 1 ? (1.f - Sdata[i]) : std::max(0.f, Sdata[i] - margin_);
      }
      return true;
        */
    }
}

pub struct CosineEmbeddingCriterionGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    margin:  f32,
}

impl<Context> CosineEmbeddingCriterionGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "margin", margin_, 0.0)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& S = Input(0);
      auto& Y = Input(1);
      auto& dOutput = Input(2);

      auto* dS = Output(0, S.sizes(), at::dtype<float>());

      const float* Sdata = S.data<float>();
      const int* Ydata = Y.data<int>();
      const float* dOutput_data = dOutput.data<float>();
      float* dSdata = dS->template mutable_data<float>();
      for (int i = 0; i < S.numel(); ++i) {
        dSdata[i] = dOutput_data[i] *
            (Ydata[i] == 1 ? -1.f : static_cast<float>(Sdata[i] >= margin_));
      }
      return true;
        */
    }
}

register_cpu_operator!{
    CosineEmbeddingCriterion,
    CosineEmbeddingCriterionOp<CPUContext>
}

register_cpu_operator!{
    CosineEmbeddingCriterionGradient,
    CosineEmbeddingCriterionGradientOp<CPUContext>
}

num_inputs!{CosineEmbeddingCriterionGradient, 3}

num_outputs!{CosineEmbeddingCriterionGradient, 1}

pub struct GetCosineEmbeddingCriterionGradient {

}

impl GetGradientDefs for GetCosineEmbeddingCriterionGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CosineEmbeddingCriterionGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{
    CosineEmbeddingCriterion,
    GetCosineEmbeddingCriterionGradient
}
