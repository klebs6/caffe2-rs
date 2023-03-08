crate::ix!();

use crate::{
    OperatorStorage,
    OperatorDef,
    CPUContext,
    APMeterOp,
    TensorShape
};

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ArgOp<Context, Reducer> {

    storage: OperatorStorage,
    context: Context,

    axis:      i32,
    reducer:   Reducer,
    keep_dims: bool,
}

impl<Context,Reducer> ArgOp<Context, Reducer> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<std::int32_t, std::int64_t, float, double>>::
            call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);

            const int ndim = X.dim();
            if (axis_ == -1) {
              axis_ = ndim - 1;
            }
            CAFFE_ENFORCE_GE(axis_, 0);
            CAFFE_ENFORCE_LT(axis_, ndim);
            const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
            std::vector<int64_t> Y_dims;
            Y_dims.reserve(ndim);
            int prev_size = 1;
            int next_size = 1;
            for (int i = 0; i < axis_; ++i) {
              Y_dims.push_back(X_dims[i]);
              prev_size *= X_dims[i];
            }
            if (keep_dims_) {
              Y_dims.push_back(1);
            }
            for (int i = axis_ + 1; i < ndim; ++i) {
              Y_dims.push_back(X_dims[i]);
              next_size *= X_dims[i];
            }
            auto* Y = Output(0, Y_dims, at::dtype<int64_t>());
            const int n = X_dims[axis_];
            return reducer_(
                prev_size,
                next_size,
                n,
                X.template data<T>(),
                Y->template mutable_data<int64_t>(),
                &context_);
        */
    }
}

#[test] fn arg_max_reducer_example() {

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ArgMax",
        ["X"],
        ["Indices"],
        axis=2,
        keepdims=False
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(3,3,3))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Indices:", workspace.FetchBlob("Indices"))

    X: [[[4. 9. 6.]
      [6. 6. 1.]
      [9. 5. 4.]]

     [[6. 7. 4.]
      [7. 9. 1.]
      [3. 2. 8.]]

     [[3. 4. 6.]
      [5. 2. 7.]
      [1. 5. 7.]]]
    Indices: [[1 0 0]
     [1 1 2]
     [2 2 2]]

    */
}

/**
  | Retrieve the argmax of an axis dimension
  | specified by the `axis` argument.
  | 
  | Given an input tensor and two arguments
  | (`axis` and `keepdims`), returns a
  | tensor containing the indices of the
  | largest element along the given axis.
  | 
  | If the `keepdims` arg is *True* (default),
  | the shape of the output tensor matches
  | the input tensor except the `axis` dimension
  | equals 1.
  | 
  | Else, the `axis` dimension of the output
  | tensor is removed.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc
  |
  */
pub struct ArgMaxReducer<Context> {
    phantom: PhantomData<Context>,

}

num_inputs!{ArgMax, 1}

num_outputs!{ArgMax, 1}

inputs!{ArgMax, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{ArgMax, 
    0 => ("Indices", "*(type: Tensor`<float>`)* Tensor of indices for the largest values.")
}

args!{ArgMax, 
    0 => ("axis", "*(type: int; default: -1)* The axis to get argmax."),
    1 => ("keepdims", "*(type: bool; default: True)* If True (default), 
        the output tensor shape will match the input tensor shape except the `axis` dimension equals 1. 
        Else, the `axis` dimension of the output tensor is removed.")
}

tensor_inference_function!{ArgMax, InferTensor }

#[test] fn arg_min_reducer_example() {
    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ArgMin",
        ["X"],
        ["Indices"],
        axis=1
    )

    workspace.FeedBlob("X", (np.random.randint(10, size=(5,5))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Indices:", workspace.FetchBlob("Indices"))

    X: [[9. 4. 6. 4. 1.]
      [5. 9. 8. 3. 4.]
      [6. 1. 0. 2. 9.]
      [7. 8. 2. 4. 9.]
      [3. 9. 4. 9. 4.]]
    Indices: [[4]
      [3]
      [2]
      [2]
      [0]]
    */
}

/**
  | Retrieve the argmin of an axis dimension
  | specified by the `axis` argument.
  | 
  | Given an input tensor and two arguments
  | (`axis` and `keepdims`), returns a
  | tensor containing the indices of the
  | smallest element along the given axis.
  | 
  | If the `keepdims` arg is *True* (default),
  | the shape of the output tensor matches
  | the input tensor except the `axis` dimension
  | equals 1.
  | 
  | Else, the `axis` dimension of the output
  | tensor is removed.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc
  |
  */
pub struct ArgMinReducer<Context> {
    phantom: PhantomData<Context>,

}

num_inputs!{ArgMin, 1}

num_outputs!{ArgMin, 1}

inputs!{ArgMin, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{ArgMin, 
    0 => ("Indices", "*(type: Tensor`<float>`)* Tensor of indices for the smallest values.")
}

args!{ArgMin, 
    0 => ("axis", "*(type: int; default: -1)* The axis to get argmin."),
    1 => ("keepdims", "*(type: bool; default: True)* If True (default), 
        the output tensor shape will match the input tensor shape except the `axis` dimension equals 1. 
        Else, the `axis` dimension of the output tensor is removed.")
}

tensor_inference_function!{ArgMin, InferTensor}

#[inline] pub fn compute_arg_impl<T, Compare, Context>(
    prev_size:  i32,
    next_size:  i32,
    n:          i32,
    comp:       &Compare,
    x:          *const T,
    y:          *mut i64,
    context:    *mut Context) 
{
    todo!();
    /*
        math::Set<int64_t, Context>(prev_size * next_size, int64_t(0), Y, context);
      for (int i = 0; i < prev_size; ++i) {
        const T* cur_X = X + i * n * next_size + next_size;
        for (int k = 1; k < n; ++k) {
          for (int j = 0; j < next_size; ++j) {
            int64_t* cur_Y = Y + i * next_size + j;
            if (comp(*cur_X, X[i * n * next_size + *cur_Y * next_size + j])) {
              *cur_Y = k;
            }
            ++cur_X;
          }
        }
      }
    */
}

impl ArgMaxReducer<CPUContext> {

    #[inline] pub fn invoke<T>(
        &mut self,
        prev_size: i32,
        next_size: i32,
        n:         i32,
        x:         *const T,
        y:         *mut i64,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            ComputeArgImpl(prev_size, next_size, n, std::greater<T>(), X, Y, context);
          return true;
        */
    }
}

impl ArgMinReducer<CPUContext> {

    #[inline] pub fn invoke<T>(
        &mut self,
        prev_size: i32,
        next_size: i32,
        n:         i32,
        x:         *const T,
        y:         *mut i64,
        context:   *mut CPUContext) -> bool 
    {
        todo!();
        /*
            ComputeArgImpl(prev_size, next_size, n, std::less<T>(), X, Y, context);
          return true;
        */
    }
}

register_cpu_operator!{ArgMax, ArgOp<CPUContext, ArgMaxReducer<CPUContext>>}

register_cpu_operator!{ArgMin, ArgOp<CPUContext, ArgMinReducer<CPUContext>>}

#[inline] pub fn infer_tensor(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<TensorShape> out(1);
      ArgumentHelper helper(def);
      int axis = helper.GetSingleArgument("axis", -1);
      const bool keep_dims = helper.GetSingleArgument("keepdims", true);
      const auto& in_dims = in[0].dims();
      auto* out_dims = out[0].mutable_dims();
      if (axis == -1) {
        axis = in_dims.size() - 1;
      }
      for (int i = 0; i < axis; ++i) {
        out_dims->Add(in_dims.Get(i));
      }
      if (keep_dims) {
        out_dims->Add(1);
      }
      for (int i = axis + 1; i < in_dims.size(); ++i) {
        out_dims->Add(in_dims.Get(i));
      }
      out[0].set_data_type(TensorProto::INT64);
      return out;
    */
}

should_not_do_gradient!{ArgMax}

should_not_do_gradient!{ArgMin}
