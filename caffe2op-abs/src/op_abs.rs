crate::ix!();

use crate::{
    CPUContext,
    GradientMakerBase,
    OperatorDef
};

#[test] fn op_abs_example() {
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Abs",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [ 0.3005476   1.551666   -1.3591481   0.39191285 -0.21866608]
    Y: [0.3005476  1.551666   1.3591481  0.39191285 0.21866608]
    */
}

/**
  | Calculates the absolute value of the
  | given input tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc
  |
  */
pub struct AbsFunctor<Context> {
    context: Context,

}

num_inputs!{Abs, 1}

num_outputs!{Abs, 1}

inputs!{Abs, 
    0 => ("X", "*(type: Tensor<float>)* Input tensor.")
}

outputs!{Abs, 
    0 => ("Y", "*(type: Tensor`<float>`)* Absolute value of input element-wise.")
}

identical_type_and_shape!{Abs}

inherit_onnx_schema!{Abs}

impl<Context> AbsFunctor<Context> {

    #[inline] pub fn invoke<T>(
        &self, 
        n: i32,
        x: *const T,
        y: *mut T,
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            math::Abs(N, X, Y, context);
            return true;
        */
    }
}

pub struct AbsGradientFunctor<Context> {
    context: Context,

}

impl AbsGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        &mut self,
        x_dims:   &Vec<i32>,
        dY_dims:  &Vec<i32>,
        x:        *const T,
        dY:       *const T,
        dX:       *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
          ConstEigenVectorArrayMap<T> dY_arr(dY, size);
          ConstEigenVectorArrayMap<T> X_arr(X, size);
          EigenVectorMap<T>(dX, size) =
              (X_arr == T(0)).select(T(0), (X_arr > T(0)).select(dY_arr, -dY_arr));
          return true;
        */
    }
}

register_cpu_operator!{
    Abs,
    UnaryElementwiseOp<TensorTypes<f32>, CPUContext, AbsFunctor<CPUContext>>
}

register_cpu_operator!{
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        AbsGradientFunctor<CPUContext>>
}

num_inputs!{AbsGradient, 2}

num_outputs!{AbsGradient, 1}

identical_type_and_shape_of_input!{AbsGradient, 0}

pub struct GetAbsGradient;

impl GetGradientDefs for GetAbsGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AbsGradient",
            "",
            std::vector<std::string>{I(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Abs, GetAbsGradient}
