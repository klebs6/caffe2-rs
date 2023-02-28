crate::ix!();

use crate::{
    OperatorStorage,
};


#[test] fn empty_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "IsEmpty",
        ["tensor"],
        ["is_empty"],
    )

    // Use a not-empty tensor
    workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("is_empty: ", workspace.FetchBlob("is_empty"),"\n")

    // Use an empty tensor
    workspace.FeedBlob("tensor", np.empty(0))
    print("tensor:\n", workspace.FetchBlob("tensor"))

    workspace.RunOperatorOnce(op)
    print("is_empty: ", workspace.FetchBlob("is_empty"))

    tensor:
     [[ 0.26018378  0.6778789 ]
     [-1.3097627  -0.40083608]]
    is_empty:  False

    tensor:
     []
    is_empty:  True
    */
}

/**
  | The *IsEmpty* op accepts a single input
  | $tensor$, and produces a single boolean
  | output $is\_empty$.
  | 
  | The output is
  | 
  | True if and only if $tensor$ has size
  | == 0.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
  |
  */
pub struct IsEmptyOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

    storage: OperatorStorage,
    context: Context,
}

impl<Context> IsEmptyOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        auto* output = Output(0, std::vector<int64_t>{}, at::dtype<bool>());
        *output->template mutable_data<bool>() = (input.numel() == 0);
        return true;
        */
    }
}

register_cpu_operator!{IsEmpty, IsEmptyOp<CPUContext>}

num_inputs!{IsEmpty, 1}

num_outputs!{IsEmpty, 1}

inputs!{IsEmpty, 
    0 => ("tensor", "Input data tensor to check if empty.")
}

outputs!{IsEmpty, 
    0 => ("is_empty", "Output scalar boolean tensor. True if input has size == 0.")
}

scalar_type!{IsEmpty, TensorProto_DataType::TensorProto_DataType_BOOL}
