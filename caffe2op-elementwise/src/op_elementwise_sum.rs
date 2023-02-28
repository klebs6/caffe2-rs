crate::ix!();

use crate::{
    OperatorStorage,
};


#[test] fn elementwise_sum_example1() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sum",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([[1,2],[3,4]]).astype(np.float32))
    workspace.FeedBlob("B", np.array([[5,6],[7,8]]).astype(np.float32))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("A"))

    **Result**

    A: [[1. 2.]
     [3. 4.]]
    B: [[5. 6.]
     [7. 8.]]
    C: [[1. 2.]
     [3. 4.]]

    */
}

#[test] fn elementwise_sum_example2() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sum",
        ["A",  "B"],
        ["A"],  // inplace
    )

    workspace.FeedBlob("A", np.array([[1,2,5],[8,3,4]]).astype(np.float32))
    workspace.FeedBlob("B", np.array([[9,5,6],[6,7,8]]).astype(np.float32))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("A after Sum:", workspace.FetchBlob("A"))

    A: [[1. 2. 5.]
     [8. 3. 4.]]
    B: [[9. 5. 6.]
     [6. 7. 8.]]
    A after Sum: [[10.  7. 11.]
     [14. 10. 12.]]

    */
}

/**
 | Element-wise sum of each of the input tensors. The
 | first input tensor can be used in-place as the
 | output tensor, in which case the sum will be done
 | in place and results will be accumulated the first
 | input tensor. All inputs and outputs must have the
 | same shape and data type.
 |
 | Github Links:
 |
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_sum_op.cc
 |
 */
pub struct ElementwiseSumOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{
    Sum, 
    ElementwiseSumOp<CPUContext>
}

num_inputs!{Sum, (1,INT_MAX)}

num_outputs!{Sum, 1}

inputs!{Sum, 
    0 => ("A", "*(type: Tensor`<float>`)* First tensor to be added element-wise."),
    1 => ("B", "*(type: Tensor`<float>`)* Second tensor to be added element-wise.")
}

outputs!{Sum, 
    0 => ("C", "*(type: Tensor`<float>`)* Sum of A and B.")
}

allow_inplace!{Sum, vec![(0, 0)]}

cost_inference_function!{Sum, CostInferenceForSum}

inputs_can_cross_devices!{Sum}

identical_type_and_shape_of_input!{Sum, 0}

inherit_onnx_schema!{Sum}
