crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
};

/**
  | Computes the element-wise negative
  | of the input.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/negative_op.cc
  |
  */
pub struct NegativeFunctor<Context> { 

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

impl<Context> NegativeFunctor<Context> {
    
    #[inline] pub fn invoke<T>(
        &mut self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Neg(N, X, Y, context);
        return true;
        */
    }
}

register_cpu_operator!{
    Negative,
    UnaryElementwiseOp<NumericTypes, CPUContext, NegativeFunctor<CPUContext>>
}

register_cuda_operator!{
    Negative,
    UnaryElementwiseOp<
        NumericTypes,
        CUDAContext,
        NegativeFunctor<CUDAContext>>
}

#[test] fn negative_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Negative",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", (np.random.rand(3,3).astype(np.float32)))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[0.83296907 0.61407167 0.32562155]
     [0.59304523 0.03111175 0.29365504]
     [0.09478621 0.5424558  0.73940724]]
    Y: [[-0.83296907 -0.61407167 -0.32562155]
     [-0.59304523 -0.03111175 -0.29365504]
     [-0.09478621 -0.5424558  -0.73940724]]
    */
}

num_inputs!{Negative, 1}

num_outputs!{Negative, 1}

inputs!{Negative, 
    0 => ("X", "*(type: Tensor`<float>`)* 1D input tensor.")
}

outputs!{Negative, 
    0 => ("Y", "*(type: Tensor`<float>`)* 1D output tensor.")
}

identical_type_and_shape!{Negative}

inherit_onnx_schema!{Negative, "Neg"}

allow_inplace!{Negative, vec![(0, 0)]}

pub struct GetNegativeGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetNegativeGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Negative",
            "",
            std::vector<std::string>{GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Negative, GetNegativeGradient}
