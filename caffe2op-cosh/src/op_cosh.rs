crate::ix!();

use crate::{
    CPUContext,
    GradientMakerBase,
    OperatorDef,
};

#[test] fn cosh_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Cosh",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.rand(5).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [0.66423494 0.32074615 0.81523746 0.90423071 0.39275789]
    Y: [1.22883528 1.05188156 1.35112322 1.43744212 1.07812598]

    */
}

/**
  | Calculates the hyperbolic cosine of
  | the given input tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosh_op.cc
  |
  */
pub struct CoshFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{Cosh, 1}

num_outputs!{Cosh, 1}

inputs!{Cosh, 
    0 => ("input", "Input tensor")
}

outputs!{Cosh, 
    0 => ("output", "The hyperbolic cosine values of the input tensor, computed element-wise")
}

identical_type_and_shape!{Cosh}

inherit_onnx_schema!{Cosh}

impl<Context> CoshFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &mut self, 
        n:         i32, 
        x:         *const T, 
        y:         *mut T, 
        context:   *mut Context) -> bool 
    {
        todo!();
        /*
            math::Cosh(N, X, Y, context);
            return true;
        */
    }
}

pub struct CoshGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{CoshGradient, 2}

num_outputs!{CoshGradient, 1}

identical_type_and_shape_of_input!{CoshGradient, 0}

impl CoshGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        &self, 
        dY_dims:   &Vec<i32>,
        x_dims:    &Vec<i32>,
        dY:        *const T,
        x:         *const T,
        dX:        *mut T,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) = dY_arr * (X_arr.exp() - (-X_arr).exp()) / 2;
          return true;
        */
    }
}

register_cpu_operator!{
    Cosh,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CoshFunctor<CPUContext>>
}

register_cpu_operator!{
    CoshGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CoshGradientFunctor<CPUContext>>
}

pub struct GetCoshGradient;

impl GetGradientDefs for GetCoshGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CoshGradient",
            "",
            std::vector<std::string>{GO(0), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Cosh, GetCoshGradient}
