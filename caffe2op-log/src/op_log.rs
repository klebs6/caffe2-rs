crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
};

/**
  | Calculates the natural log of the given
  | input tensor ($ln(x)$), element-wise.
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log_op.cc
  |
  */
pub struct LogFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Log, 1}

num_outputs!{Log, 1}

inputs!{Log, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Log, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor computed as the natural log of the input tensor computed, element-wise.")
}

identical_type_and_shape!{Log}

allow_inplace!{Log, vec![(0, 0)]}

inherit_onnx_schema!{Log}

impl<Context> LogFunctor<Context> {

    #[inline] pub fn invoke<T>(&mut self, 
        n: i32, 
        x: *const T, 
        y: *mut T, 
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Log(N, X, Y, context);
            return true;
        */
    }
}

register_cpu_operator!{
    Log,
    UnaryElementwiseOp<
    TensorTypes<f32>, 
    CPUContext, 
    LogFunctor<CPUContext>>
}

#[test] fn log_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Log",
        ["X"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32))
    print("X before running op:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("X after running op:", workspace.FetchBlob("X"))

    X before running op:
    [[0.07341351 0.15404125 0.386613  ]
     [0.34090295 0.99727786 0.24141751]
     [0.32016268 0.8724168  0.93515724]]
    X after running op:
    [[-2.6116474  -1.8705349  -0.9503311 ]
     [-1.0761575  -0.00272586 -1.4212275 ]
     [-1.138926   -0.13648799 -0.06704059]]
    */
}

pub struct GetLogGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLogGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "Div",
            "",
            std::vector<std::string>{GO(0), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Log, GetLogGradient}

register_cuda_operator!{
    Log,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CUDAContext,
        LogFunctor<CUDAContext>>
}
