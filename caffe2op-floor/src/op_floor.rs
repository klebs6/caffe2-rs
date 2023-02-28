crate::ix!();

use crate::{
    OperatorStorage,
};

/**
  | Element-wise application of the floor
  | function ($y=floor(x)$) to the input
  | tensor `X`. Output tensor shape is the
  | same as the input tensor. This operator
  | can be used in an in-place fashion by
  | using the same input blob as the output
  | blob.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/floor_op.cc
  |
  */
pub struct FloorOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    phantom: PhantomData<T>,
}

num_inputs!{Floor, 1}

num_outputs!{Floor, 1}

inputs!{Floor, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Floor, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

allow_inplace!{Floor, vec![(0, 0)]}

impl<T, Context> FloorOp<T, Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        auto* Y = Output(0, X.sizes(), at::dtype<float>());

        const float* Xdata = X.template data<float>();
        float* Ydata = Y->template mutable_data<float>();
        for (int i = 0; i < X.numel(); ++i) {
          Ydata[i] = std::floor(Xdata[i]);
        }
        return true;
        */
    }
}

register_cpu_operator!{
    Floor, 
    FloorOp<f32, CPUContext>
}

#[test] fn floor_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Floor",
        ["X"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.uniform(-10, 10, (5,5))).astype(np.float32))
    print("X before running op:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("X after running op:", workspace.FetchBlob("X"))

    X before running op:
    [[ 3.813361   -1.319647    5.2089314  -4.931328    0.6218652 ]
     [ 7.2757645   5.5552588   5.785643   -2.4790506  -0.41400087]
     [ 1.1541046  -6.933266    3.3754056   1.6569928  -1.7670316 ]
     [-3.4932013   4.891472    1.5530115  -3.2443287  -4.605099  ]
     [-4.574543   -7.360948    5.91305    -8.196495   -5.357458  ]]
    X after running op:
    [[ 3. -2.  5. -5.  0.]
     [ 7.  5.  5. -3. -1.]
     [ 1. -7.  3.  1. -2.]
     [-4.  4.  1. -4. -5.]
     [-5. -8.  5. -9. -6.]]
    */
}

// TODO: Write gradient for this when needed
gradient_not_implemented_yet!{Floor}
