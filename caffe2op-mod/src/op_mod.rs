crate::ix!();

use crate::{
    OperatorStorage,
    CPUContext,
};

/**
  | Element-wise modulo operation. Each
  | element in the output is the modulo result
  | of the corresponding element in the
  | input data. The divisor of the modulo
  | is provided by the `divisor` argument.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mod_op.cc
  |
  */
pub struct ModOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    divisor:             i64,
    sign_follow_divisor: bool,
}

register_cpu_operator!{Mod, ModOp<CPUContext>}

num_inputs!{Mod, 1}

num_outputs!{Mod, 1}

inputs!{Mod, 
    0 => ("X", "*(type: Tensor`<int>`)* Input tensor with int32 or int64 data.")
}

outputs!{Mod, 
    0 => ("Y", "*(type: Tensor`<int>`)* Output tensor of data with modulo operation applied.")
}

args!{Mod, 
    0 => ("divisor", "*(type: int; default: 0)* Divisor of the modulo operation (must be >= 1)."),
    1 => ("sign_follow_divisor", "*(type: bool; default: False)* If true, sign of output matches divisor, else if false, sign follows dividend.")
}

identical_type_and_shape!{Mod}

allow_inplace!{Mod, vec![(0, 0)]}

should_not_do_gradient!{ModOp}

#[test] fn mod_op_example() {

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Mod",
        ["X"],
        ["Y"],
        divisor=10
    )

    workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))))
    print("X before running op:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("X after running op:", workspace.FetchBlob("Y"))

    X before running op:
    [[56 22 43 13 60]
     [ 4 55 58 10 45]
     [64 66  4  3 66]
     [10 36 47 52 78]
     [91  4 36 47 95]]
    X after running op:
    [[6 2 3 3 0]
     [4 5 8 0 5]
     [4 6 4 3 6]
     [0 6 7 2 8]
     [1 4 6 7 5]]

    */
}

input_tags!{
    ModOp {
        Data
    }
}

impl<Context> ModOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        divisor_ = this->template GetSingleArgument<int64_t>("divisor", 0);
        CAFFE_ENFORCE_NE(divisor_, 0, "divisor must not be 0");
        sign_follow_divisor_ =
            this->template GetSingleArgument<bool>("sign_follow_divisor", false);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(DATA));
        */
    }
}

impl ModOp<CPUContext> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
      auto N = data.numel();
      const auto* data_ptr = data.template data<T>();

      auto* output = Output(0, Input(DATA).sizes(), at::dtype<T>());
      auto* output_ptr = output->template mutable_data<T>();

      for (auto i = 0; i < N; i++) {
        output_ptr[i] = data_ptr[i] % divisor_;
        if (output_ptr[i] && sign_follow_divisor_ &&
            ((output_ptr[i] > 0) != (divisor_ > 0))) {
          output_ptr[i] += divisor_;
        }
      }
      return true;
        */
    }
}
