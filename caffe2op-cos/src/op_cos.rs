crate::ix!();

use crate::{
    GradientMakerBase,
    CPUContext,
    OperatorDef,
};

#[test] fn cos_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Cos",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.rand(5).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [0.6816719  0.76771533 0.933932   0.01404487 0.11862425]
    Y: [0.7765203  0.71949923 0.5946774  0.99990135 0.9929724 ]
    */
}

/**
  | Calculates the cosine of the given input
  | tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cos_op.cc
  |
  */
pub struct CosFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> CosFunctor<Context> {

    #[inline] pub fn invoke<T>(
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cos(N, X, Y, context);
            return true;
        */
    }
}

pub struct CosGradientFunctor<Context> { 
    phantom: PhantomData<Context>,
}

impl CosGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        x_dims:  &Vec<i32>,
        dY_dims: &Vec<i32>,
        x:       *const T,
        dY:      *const T,
        dX:      *mut T,
        context: *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) = -dY_arr * X_arr.sin();
          return true;
        */
    }
}

register_cpu_operator!{
    Cos,
    UnaryElementwiseOp<
        TensorTypes<f32>, 
        CPUContext, 
        CosFunctor<CPUContext>
    >
}

register_cpu_operator!{
    CosGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CosGradientFunctor<CPUContext>>
}

num_inputs!{Cos,  1}

num_outputs!{Cos, 1}

inputs!{Cos, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Cos, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the cosine of the input tensor, element-wise.")
}

identical_type_and_shape!{Cos}

num_inputs!{CosGradient, 2}

num_outputs!{CosGradient, 1}

identical_type_and_shape!{CosGradient}

pub struct GetCosGradient {}

impl GetGradientDefs for GetCosGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CosGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Cos, GetCosGradient}
