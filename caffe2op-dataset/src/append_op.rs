crate::ix!();

/**
  | Append input `B` to the end of input `A`.
  | 
  | - It is required that this operation
  | run in-place, meaning that the input
  | `A` blob must match the output blob.
  | 
  | - All except the outer-most dimension
  | must be the same between `A` and `B`.
  | 
  | - Input `A` may have to be re-allocated
  | in order for accommodate to the new size.
  | Currently, an exponential growth ratio
  | is used in order to ensure amortized
  | constant time complexity.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AppendOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{Append, 2}

num_outputs!{Append, 1}

inputs!{Append, 
    0 => ("A", "(*Tensor*): base input tensor of shape $(N, d_1, d_2, ..., d_n)$"),
    1 => ("B", "(*Tensor*): second input tensor of shape $(M, d_1, d_2, ..., d_n)$ to be appended to the base")
}

outputs!{Append, 
    0 => ("A", "(*Tensor*): output tensor of shape $(N+M, d_1, d_2, ..., d_n)$")
}

enforce_inplace!{Append, vec![(0, 0)]}

impl<Context> AppendOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& a = Input(0);
        auto& b = Input(1);
        auto* c = Output(0);
        CAFFE_ENFORCE(b.dim() >= 1);
        if (a.numel() == 0 && a.size(0) == 0) {
          c->CopyFrom(b);
          return true;
        }
        CAFFE_ENFORCE(&a == c, "First argument must be in-place.");
        CAFFE_ENFORCE(c->dim() == b.dim());
        CAFFE_ENFORCE(b.dim() == c->dim());
        CAFFE_ENFORCE(a.dtype() == b.dtype());
        for (int i = 1; i < a.dim(); ++i) {
          CAFFE_ENFORCE(a.sizes()[i] == b.sizes()[i]);
        }
        auto oldSize = c->numel();
        c->Extend(b.sizes()[0], kDatasetGrowthPct);
        auto* dst = (char*)c->raw_mutable_data() + oldSize * b.dtype().itemsize();
        context_.CopyItemsSameDevice(b.dtype(), b.numel(), b.raw_data(), dst);
        return true;
        */
    }
}

#[test] fn append_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Append",
        ["A", "B"],
        ["A"],
    )

    workspace.FeedBlob("A", np.random.randint(10, size=(1,3,3)))
    workspace.FeedBlob("B", np.random.randint(10, size=(2,3,3)))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("A:", workspace.FetchBlob("A"))

    A:
    [[[3 8 7]
      [1 6 6]
      [5 0 6]]]
    B:
    [[[4 3 1]
      [7 9 6]
      [9 4 5]]

     [[7 7 4]
      [9 8 7]
      [1 6 6]]]
    A:
    [[[3 8 7]
      [1 6 6]
      [5 0 6]]

     [[4 3 1]
      [7 9 6]
      [9 4 5]]

     [[7 7 4]
      [9 8 7]
      [1 6 6]]]

    */
}

