crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
};

#[test] fn assert_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Assert",
        ["A"],
        [],
        error_msg="Failed assertion from Assert operator"
    )

    workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.int32))
    print("A:", workspace.FetchBlob("A"))
    try:
        workspace.RunOperatorOnce(op)
    except RuntimeError:
        print("Assertion Failed!")
    else:
        print("Assertion Passed!")

    A:
    [[7 5 6]
     [1 2 4]
     [5 3 7]]
    Assertion Passed!

    */
}

/**
| Takes in a tensor of type *bool*, *int*, *long*,
| or *long long* and checks if all values are True
| when coerced into a boolean. In other words, for
| non-bool types this asserts that all values in the
| tensor are non-zero. 
|
| If a value is False after coerced into a boolean,
| the operator throws an error. 
|
| Else, if all values are True, nothing is
| returned. For tracability, a custom error message
| can be set using the `error_msg` argument.
|
| Github Links:
| - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/assert_op.cc
|
*/
pub struct AssertOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:    OperatorStorage,
    context:    Context,

    cmp_tensor: Tensor, //{CPU};
    error_msg:  String,
}

num_inputs!{Assert, 1}

num_outputs!{Assert, 0}

inputs!{Assert, 
    0 => ("X", "(*Tensor*): input tensor")
}

args!{Assert, 
    0 => ("error_msg", "(*string*): custom error message to be thrown when the input does not pass assertion")
}

impl<Context> AssertOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            error_msg_(
                this->template GetSingleArgument<std::string>("error_msg", ""))
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            // Copy into CPU context for comparison
            cmp_tensor_.CopyFrom(Input(0));
            auto* cmp_data = cmp_tensor_.template data<T>();

            for (int64_t i = 0; i < cmp_tensor_.numel(); ++i) {
              CAFFE_ENFORCE((bool)cmp_data[i], [&]() {
                std::stringstream ss;
                ss << "Assert failed for element " << i
                   << " in tensor, value: " << cmp_data[i] << "\n";
                if (!error_msg_.empty()) {
                  ss << "Error message: " << error_msg_;
                }
                return ss.str();
              }());
            }
            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<long, int, bool>>::call(this, Input(0));
        */
    }
}

register_cpu_operator!{
    Assert, 
    AssertOp<CPUContext>
}
