crate::ix!();

use crate::{
    GradientMakerBase,
    CPUContext,
    OperatorDef
};


///--------------------
pub struct SinhFunctor<Context> {
    
    phantom: PhantomData<Context>,
}
impl<Context> SinhFunctor<Context> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Sinh(N, X, Y, context);
        return true;
        */
    }
}

///-------------------
pub struct SinhGradientFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

impl SinhGradientFunctor<CPUContext> {
    
    #[inline] pub fn forward<T>(&self, 
        dy_dims: &Vec<i32>,
        x_dims:  &Vec<i32>,
        dy:      *const T,
        x:       *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const int size = std::accumulate(
          X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
      ConstEigenVectorArrayMap<T> dY_arr(dY, size);
      ConstEigenVectorArrayMap<T> X_arr(X, size);
      EigenVectorMap<T>(dX, size) = dY_arr * (X_arr.exp() + (-X_arr).exp()) / 2;
      return true;
        */
    }
}

///----------------------------
/**
Calculates the hyperbolic sine of the given input tensor, element-wise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinh_op.cc

*/
register_cpu_operator!{Sinh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinhFunctor<CPUContext>>}

num_inputs!{Sinh, 1}

num_outputs!{Sinh, 1}

inputs!{Sinh, 
    0 => ("input", "Input tensor")
}

outputs!{Sinh, 
    0 => ("output", "The hyperbolic sine values of the input tensor, computed element-wise")
}

identical_type_and_shape!{Sinh}

inherit_onnx_schema!{Sinh}

#[test] fn sinh_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sinh",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.rand(5).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    ```

    **Result**

    ```

    X: [0.98907769 0.52907848 0.03216429 0.94983935 0.47881418]
    Y: [1.15841695 0.5541099  0.03216984 1.09924557 0.49732079]
    */
}

///------------------------------
register_cpu_operator!{SinhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinhGradientFunctor<CPUContext>>}

num_inputs!{SinhGradient, 2}

num_outputs!{SinhGradient, 1}

identical_type_and_shape_of_input!{SinhGradient, 0}

pub struct GetSinhGradient {
    base: GradientMakerBase,
}


impl GetSinhGradient {
    
    #[inline] pub fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SinhGradient",
            "",
            std::vector<std::string>{GO(0), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Sinh, GetSinhGradient}
