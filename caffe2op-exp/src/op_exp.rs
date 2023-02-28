crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
};

#[test] fn exp_functor_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Exp",
        ["X"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
    print("X before running op:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("X after running op:", workspace.FetchBlob("X"))

    X before running op:
    [[0.5821691  0.07719802 0.50159824]
     [0.40952456 0.36788362 0.84887683]
     [0.02472685 0.65730894 0.9066397 ]]

    X after running op:
    [[1.7899168 1.080256  1.6513585]
     [1.5061016 1.4446739 2.3370204]
     [1.0250351 1.9295927 2.4759884]]
    */
}

/**
  | Calculates the exponential of the given
  | input tensor ($exp(x)$), element-wise.
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/exp_op.cc
  |
  */
pub struct ExpFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> ExpFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self,
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Exp(N, X, Y, context);
            return true;
        */
    }
}

register_cpu_operator!{
    Exp,
    UnaryElementwiseOp<TensorTypes<f32>, CPUContext, ExpFunctor<CPUContext>>
}

num_inputs!{Exp, 1}

num_outputs!{Exp, 1}

inputs!{Exp, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Exp, 
    0 => ("Y", "*(type: Tensor`<float>`)* The exponential of the input tensor computed element-wise.")
}

identical_type_and_shape!{Exp}

inherit_onnx_schema!{Exp}

allow_inplace!{Exp, vec![(0, 0)]}

pub struct GetExpGradient;

impl GetGradientDefs for GetExpGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Mul",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Exp, GetExpGradient}

register_cuda_operator!{
    Exp,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CUDAContext,
        ExpFunctor<CUDAContext>>
}
