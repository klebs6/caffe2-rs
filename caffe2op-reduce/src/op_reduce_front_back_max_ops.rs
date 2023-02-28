crate::ix!();

use crate::{
    OperatorStorage,
    OperatorDef,
    CPUContext,
    GradientMakerBase
};

pub struct MaxReduceDimsOp<T,Context,const FIRSTDIMS: bool> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    num_reduce_dims:  i32,
    phantom: PhantomData<T>,
}

impl<T,Context,const FIRSTDIMS: bool> MaxReduceDimsOp<T,Context,FIRSTDIMS> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            num_reduce_dims_( this->template GetSingleArgument<int32_t>("num_reduce_dim", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        CAFFE_ENFORCE(
            num_reduce_dims_ >= 0 && num_reduce_dims_ <= X.dim(),
            "For N-dim input tensor, support num_reduce_dims in range [0, N].");

        const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                                   : X.size_to_dim(X.dim() - num_reduce_dims_);
        const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                                   : X.size_from_dim(X.dim() - num_reduce_dims_);

        vector<int64_t> output_shape;
        int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
        int end_index = FIRSTDIMS ? X.dim() : X.dim() - num_reduce_dims_;

        for (int i = start_index; i < end_index; ++i) {
          output_shape.push_back(X.sizes()[i]);
        }
        auto* Y = Output(0, output_shape, at::dtype<float>());
        float* out_data = Y->template mutable_data<float>();

        if (cols == 0 || rows == 0) {
          math::Set(Y->numel(), static_cast<float>(0), out_data, &context_);
          return true;
        }

        const int32_t* lengths_data = nullptr;
        if (InputSize() > 1) {
          const auto& lengths = Input(1);
          lengths_data = lengths.template data<int32_t>();
          CAFFE_ENFORCE(
              num_reduce_dims_ == 1,
              "Given lengths input, the number of reduce dimensions should be one.");
          const int batch_size = FIRSTDIMS ? cols : rows;
          CAFFE_ENFORCE(
              lengths.numel() == batch_size,
              "The size of lengths vector doesn't match the batch size.");
        }

        const float* data = X.template data<float>();
        Compute(rows, cols, data, lengths_data, out_data);
        return true;
        */
    }
}

///----------------------------------------------
pub struct MaxReduceDimsGradientOp<T,Context,const FIRSTDIMS: bool> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:         OperatorStorage,
    context:         Context,

    num_reduce_dims: i32,
    phantom:         PhantomData<T>,
}

impl<T,Context,const FIRSTDIMS: bool> MaxReduceDimsGradientOp<T,Context,FIRSTDIMS> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            num_reduce_dims_( this->template GetSingleArgument<int32_t>("num_reduce_dim", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);
        auto& X = Input(1);
        auto& Y = Input(2);

        auto* dX = Output(0, X.sizes(), at::dtype<float>());
        const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                                   : X.size_to_dim(X.dim() - num_reduce_dims_);
        const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                                   : X.size_from_dim(X.dim() - num_reduce_dims_);

        const float* dYdata = dY.template data<float>();
        const float* Xdata = X.template data<float>();
        const float* Ydata = Y.template data<float>();

        const int32_t* lengths_data = nullptr;
        if (InputSize() > 3) {
          const auto& lengths = Input(3);
          lengths_data = lengths.template data<int32_t>();
          CAFFE_ENFORCE(
              num_reduce_dims_ == 1,
              "Given lengths input, the number of reduce dimensions should be one.");
          const int batch_size = FIRSTDIMS ? cols : rows;
          CAFFE_ENFORCE(
              lengths.numel() == batch_size,
              "The size of lengths vector doesn't match the batch size.");
        }

        float* dXdata = dX->template mutable_data<float>();
        Compute(rows, cols, dYdata, Xdata, Ydata, lengths_data, dXdata);
        return true;
        */
    }
}

#[macro_export] macro_rules! reduction_op_shape_inference {
    ($is_front_reducer:ident) => {
        todo!();
        /*
        
          CAFFE_ENFORCE_LE(1, in.size());                                           
          CAFFE_ENFORCE_GE(2, in.size());                                           
          ArgumentHelper helper(def);                                               
          int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1); 
          int start_index = is_front_reducer ? num_reduce_dims : 0;                 
          int end_index = is_front_reducer ? in[0].dims_size()                      
                                           : in[0].dims_size() - num_reduce_dims;   
          vector<int> output_shape;                                                 
          for (int i = start_index; i < end_index; ++i) {                           
            output_shape.push_back(in[0].dims(i));                                  
          }                                                                         
          return vector<TensorShape>{                                               
              CreateTensorShape(output_shape, in[0].data_type())};
        */
    }
}

// ReduceFrontMax
impl MaxReduceDimsOp<f32, CPUContext, true> {

    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        data:         *const f32,
        lengths_data: *const i32,
        out_data:     *mut f32)  {

        todo!();
        /*
            for (int i = 0; i < cols; i++) {
        float mx = data[i];
        int length = lengths_data == nullptr ? rows : lengths_data[i];
        for (int j = 1; j < length; j++) {
          mx = std::max(mx, data[j * cols + i]);
        }
        out_data[i] = mx;
      }
        */
    }
}

// ReduceBackMax
impl MaxReduceDimsOp<f32, CPUContext, false> {
    
    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        data:         *const f32,
        lengths_data: *const i32,
        out_data:     *mut f32)  {

        todo!();
        /*
            for (int i = 0; i < rows; i++) {
        float mx = data[i * cols];
        int length = lengths_data == nullptr ? cols : lengths_data[i];
        for (int j = 1; j < length; j++) {
          mx = std::max(mx, data[i * cols + j]);
        }
        out_data[i] = mx;
      }
        */
    }
}

// ReduceFrontMaxGradient
impl MaxReduceDimsGradientOp<f32, CPUContext, true> {

    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const f32,
        xdata:        *const f32,
        ydata:        *const f32,
        lengths_data: *const i32,
        d_xdata:      *mut f32)  {
        
        todo!();
        /*
            int len = cols * rows;
      for (int i = 0; i < len; i++) {
        int col = i % cols;
        int row = i / cols;
        if (lengths_data != nullptr && row >= lengths_data[col]) {
          dXdata[i] = 0.0f;
        } else {
          dXdata[i] = Xdata[i] == Ydata[col] ? dYdata[col] : 0.0f;
        }
      }
        */
    }
}

// ReduceBackMaxGradient
impl MaxReduceDimsGradientOp<f32, CPUContext, false> {

    #[inline] pub fn compute_f32_on_cpu(&mut self, 
        rows:         i32,
        cols:         i32,
        d_ydata:      *const f32,
        xdata:        *const f32,
        ydata:        *const f32,
        lengths_data: *const i32,
        d_xdata:      *mut f32)  {

        todo!();
        /*
            int len = cols * rows;
      for (int i = 0; i < len; i++) {
        int row = i / cols;
        int col = i % cols;
        if (lengths_data == nullptr || col < lengths_data[row]) {
          dXdata[i] = Xdata[i] == Ydata[row] ? dYdata[row] : 0.0f;
        } else {
          dXdata[i] = 0.0f;
        }
      }
        */
    }
}


///-----------------------------
#[test] fn reduce_front_max_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceFrontMax",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[2. 8. 1.]
      [9. 6. 6.]
      [7. 7. 0.]]

     [[4. 3. 9.]
      [9. 2. 7.]
      [6. 4. 7.]]]
    Y: [9. 8. 9.]
    */
}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **max**.
  | 
  | Can reduce more than one of the "first"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the max operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_1 * d_2 * ... * d_{n})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{0}$ dimension.
  | 
  | For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1,2]$, then $Y =
  | [max(1,4), max(5,1,7), max(2), max(9,2)]
  | = [4, 7, 2, 9]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_max_ops.cc
  |
  */
register_cpu_operator!{ReduceFrontMax, MaxReduceDimsOp<float, CPUContext, true>}

num_inputs!{ReduceFrontMax, (1,2)}

num_outputs!{ReduceFrontMax, 1}

inputs!{ReduceFrontMax, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceFrontMax, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceFrontMax, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceFrontMax, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
        REDUCTION_OP_SHAPE_INFERENCE(true)
        */
    }
}

register_cpu_operator!{ReduceFrontMaxGradient, MaxReduceDimsGradientOp<float, CPUContext, true>}

num_inputs!{ReduceFrontMaxGradient, (3,4)}

num_outputs!{ReduceFrontMaxGradient, 1}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **max**.
  | 
  | Can reduce more than one of the "last"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the max operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_0 * d_1 * d_2 * ... * d_{n-1})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{n-1}$ dimension.
  | 
  | For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1]$, then $Y = [max(1,5),
  | max(4,1,8), max(2)] = [5, 8, 2]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_max_ops.cc
  |
  */
#[test] fn reduce_back_max_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceBackMax",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[2. 5. 1.]
       [6. 1. 9.]
       [8. 5. 9.]]

      [[5. 7. 8.]
       [9. 9. 6.]
       [6. 5. 0.]]]]
    Y: [[9. 9.]]

    */
}

register_cpu_operator!{ReduceBackMax, MaxReduceDimsOp<float, CPUContext, false>}

num_inputs!{ReduceBackMax, (1,2)}

num_outputs!{ReduceBackMax, 1}

inputs!{ReduceBackMax, 
    0 => ("X", "(*Tensor`<float>`*): input tensor"),
    1 => ("lengths", "(*Tensor`<int>`*): number of elements in each sample")
}

outputs!{ReduceBackMax, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceBackMax, 
    0 => ("num_reduce_dims", "(*int*): number of dimensions to reduce (default=1)")
}

tensor_inference_function!{ReduceBackMax, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
           REDUCTION_OP_SHAPE_INFERENCE(false)
           */
    }
}

///-----------------------------
register_cpu_operator!{ReduceBackMaxGradient,  MaxReduceDimsGradientOp<float, CPUContext, false>}

num_inputs!{ReduceBackMaxGradient, (3,4)}

num_outputs!{ReduceBackMaxGradient, 1}

///-------------------------------
pub struct GetReduceFrontMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0), O(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceFrontMaxGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceFrontMax, GetReduceFrontMaxGradient}

///-------------------------------
pub struct GetReduceBackMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceBackMaxGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_in = {GO(0), I(0), O(0)};
        if (def_.input_size() == 2) {
          grad_in.push_back(I(1));
        }
        return SingleGradientDef(
            "ReduceBackMaxGradient", "", grad_in, vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceBackMax, GetReduceBackMaxGradient}
