crate::ix!();

/**
 | Computes and outputs the dot product of the two
 | input float tensors `X` and `Y`.
 |
 | Note that `X` and `Y` must be either 1D or 2D, and
 | they must be the same shape.
 |
 | The output tensor is 1D, which represents either
 | the product of each element in a respective
 | dimension if the inputs are 1D, or the sum of the
 | products in a given dimension if the inputs are 2D
 | matrices. Note that the actual dot product is
 | a scalar value, which is effectively the sum of
 | the elements in the 1D output tensor.
 |
 | For 1D inputs:
 | Given two vectors 
 | $X = [x_0, x_1, x_2]$ 
 | and $Y = [y_0, y_1, y_2]$; 
 | $Z = [x_0 * y_0, x_1 * y_1, x_2 * y_2]$
 |
 | For 2D inputs:
 | Given two matrices:
 | $$X = [
 | [x_0^0, x_1^0, x_2^0], 
 | \\ [x_0^1, x_1^1, x_2^1], 
 | \\ [x_0^2, x_1^2, x_2^2], 
 | \\ ..., 
 | \\ [x_0^n, x_1^n, x_2^n]
 | ]$$
 |
 | and
 |
 | $$Y = [
 | [y_0^0, y_1^0, y_2^0], 
 | \\ [y_0^1, y_1^1, y_2^1], 
 | \\ [y_0^2, y_1^2, y_2^2], 
 | \\ ..., 
 | \\ [y_0^n, y_1^n, y_2^n]
 | ]$$
 |
 | then
 |
 | $$Z =  \biggl[
 | \Big((x_0^0 * y_0^0) + (x_1^0 * y_1^0) + (x_2^0 * y_2^0)\Big), 
 | \\ \Big((x_0^1 * y_0^1) + (x_1^1 * y_1^1) + (x_2^1 * y_2^1)\Big), 
 | \\ \Big((x_0^2 * y_0^2) + (x_1^2 * y_1^2) + (x_2^2 * y_2^2)\Big), 
 | \\ ..., 
 | \\ \Big((x_0^n * y_0^n) + (x_1^n * y_1^n) + (x_2^n * y_2^n)\Big)\biggr]$$
 |
 | Github Link:
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc
 |
 */
pub struct DotProductOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    phantom: PhantomData<T>,
}

register_cpu_operator!{DotProduct, DotProductOp<float, CPUContext>}

num_inputs!{DotProduct, 2}

num_outputs!{DotProduct, 1}

inputs!{DotProduct, 
    0 => ("X", "*(type: Tensor`<float>`)* 1D or 2D input tensor."),
    1 => ("Y", "*(type: Tensor`<float>`)* 1D or 2D input tensor (must have the same shape as X).")
}

outputs!{DotProduct, 
    0 => ("Z", "*(type: Tensor`<float>`)* 1D output tensor.")
}

identical_type_and_shape_of_input_dim!{DotProduct, (0, 0)}

inherit_onnx_schema!{DotProduct}

cost_inference_function!{DotProduct, /* OpSchema::CostInferenceFunctionType(CostInferenceForDotProduct) */ }

tensor_inference_function!{DotProduct, TensorInferenceForDotProduct}

impl<T,Context> DotProductOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

input_tags!{
    DotProductOp {
        XIn,
        YIn
    }
}

output_tags!{
    DotProductOp {
        DotOut
    }
}

