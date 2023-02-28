crate::ix!();

use crate::{
    OperatorStorage,
    IValue,
    GradientMakerBase,
    OperatorDef,
    FunctionSchema,
    Workspace,
    Tensor,
};

/**
  | Sums the elements of the input tensor.
  | Tensor type must be float32.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc
  |
  */
pub struct SumElementsOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:  OperatorStorage,
    context:  Context,

    average:  bool,
    scratch:  Tensor, // {Context::GetDeviceType()};
    phantom:  PhantomData<T>,
}

num_inputs!{SumElements, 1}

num_outputs!{SumElements, 1}

inputs!{SumElements, 
    0 => ("X", "(*Tensor`<float>`*): blob pointing to an instance of a counter")
}

outputs!{SumElements, 
    0 => ("sum", "(*Tensor`<float>`*): Scalar tensor containing the sum (or average)")
}

args!{SumElements, 
    0 => ("average", "(*bool*): set to True to compute the average of the elements rather than the sum")
}

scalar_type!{SumElements, TensorProto::FLOAT}

#[test] fn sum_elements_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    sum_op = core.CreateOperator(
        "SumElements",
        ["X"],
        ["Y"]
    )

    avg_op = core.CreateOperator(
        "SumElements",
        ["X"],
        ["Y"],
        average=True
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(3,3)).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(sum_op)
    print("Y (sum_op):", workspace.FetchBlob("Y"))
    workspace.RunOperatorOnce(avg_op)
    print("Y (avg_op):", workspace.FetchBlob("Y"))

    X:
     [[7. 2. 5.]
     [9. 4. 2.]
     [1. 2. 5.]]
    Y (sum_op): 37.0
    Y (avg_op): 4.111111
    */
}

#[cfg(not(all(not(caffe2_is_xplat_build),not(c10_mobile))))]
impl<T,Context> SumElementsOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            average_(this->template GetSingleArgument<bool>("average", false))
        */
    }
    
    pub fn new_with_avg(
        operator_def: &OperatorDef,
        ws:           *mut Workspace,
        average:      bool) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws), average_(average)
        */
    }
}

#[cfg(all(not(caffe2_is_xplat_build),not(c10_mobile)))]
impl<T,Context> SumElementsOp<T,Context> {
    
    pub fn new(
    schema:  &FunctionSchema,
    inputs:  Vec<IValue>,
    outputs: Vec<*mut IValue>) -> Self {
    
        todo!();
        /*
            : Operator<Context>(schema, std::move(inputs), std::move(outputs)),
            average_(this->template GetSingleArgument<bool>("average", false))
        */
    }
    
    pub fn new_with_avg(
        schema:  &FunctionSchema,
        inputs:  Vec<IValue>,
        outputs: Vec<*mut IValue>,
        average: bool) -> Self {

        todo!();
        /*
            : Operator<Context>(schema, std::move(inputs), std::move(outputs)), average_(average)
        */
    }
}

impl<T,Context> SumElementsOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());

        T* data = sum->template mutable_data<T>();

        math::Sum<T, Context>(
            X.numel(), X.template data<T>(), data, &context_, &scratch_);
        if (average_ && X.numel() > 0) {
          math::Scale<float, T, Context>(
              1,
              static_cast<T>(1.) / X.numel(),
              sum->template data<T>(),
              data,
              &context_);
        }
        return true;
        */
    }
}

///--------------------------

///Sums the integer elements of the input tensor.
pub struct SumElementsIntOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    scratch:  Tensor, // {Context::GetDeviceType()};
    phantom: PhantomData<T>,
}

num_inputs!{SumElementsInt, 1}

num_outputs!{SumElementsInt, 1}

inputs!{SumElementsInt, 
    0 => ("X", "Tensor to sum up")
}

outputs!{SumElementsInt, 
    0 => ("sum", "Scalar sum")
}

scalar_type!{SumElementsInt, TensorProto::INT32}

should_not_do_gradient!{SumElementsInt}

impl<T,Context> SumElementsIntOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());
        T* data = sum->template mutable_data<T>();
        math::Sum<T, Context>(
            X.numel(), X.template data<T>(), data, &context_, &scratch_);
        return true;
        */
    }
}

///----------------------------
pub struct SumElementsGradientOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    average:  bool,
    phantom: PhantomData<T>,
}

num_inputs!{SumElementsGradient, 2}

num_outputs!{SumElementsGradient, 1}

#[cfg(not(all(not(caffe2_is_xplat_build),not(c10_mobile))))]
impl<T,Context> SumElementsGradientOp<T,Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            average_(this->template GetSingleArgument<bool>("average", false))
        */
    }
    
    pub fn new_with_avg(
        operator_def: &OperatorDef,
        ws:           *mut Workspace,
        average:      bool) -> Self {

        todo!();
        /*
            : Operator<Context>(operator_def, ws), average_(average)
        */
    }
}

#[cfg(all(not(caffe2_is_xplat_build),not(c10_mobile)))]
impl<T,Context> SumElementsGradientOp<T,Context> {
    
    pub fn new(
        schema:  &FunctionSchema,
        inputs:  Vec<IValue>,
        outputs: Vec<*mut IValue>) -> Self {
    
        todo!();
        /*
            : Operator<Context>(schema, std::move(inputs), std::move(outputs)),
            average_(this->template GetSingleArgument<bool>("average", false))
        */
    }
    
    pub fn new_with_avg(
        schema:  &FunctionSchema,
        inputs:  Vec<IValue>,
        outputs: Vec<*mut IValue>,
        average: bool) -> Self {
    
        todo!();
        /*
            : Operator<Context>(schema, std::move(inputs), std::move(outputs)), average_(average)
        */
    }
}

///-----------------------------
///Sums the squares elements of the input tensor.
pub struct SumSqrElementsOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    //USE_SIMPLE_CTOR_DTOR(SumSqrElementsOp)
    storage: OperatorStorage,
    context: Context,

    scratch:  Tensor, // {Context::GetDeviceType()};
}

num_inputs!{SumSqrElements, 1}

num_outputs!{SumSqrElements, 1}

inputs!{SumSqrElements, 
    0 => ("X", "Tensor to sum up")
}

outputs!{SumSqrElements, 
    0 => ("sum", "Scalar sum of squares")
}

args!{SumSqrElements, 
    0 => ("average", "whether to average or not")
}

scalar_type!{SumSqrElements, TensorProto::FLOAT}

impl<Context> SumSqrElementsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            bool average = this->template GetSingleArgument<bool>("average", false);
        auto& X = Input(0);

        auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());
        math::SumSqr<T, Context>(
            X.numel(),
            X.template data<T>(),
            sum->template mutable_data<T>(),
            &context_,
            &scratch_);
        if (average && X.numel() > 0) {
          math::Scale<float, T, Context>(
              1,
              float(1.) / X.numel(),
              sum->template data<T>(),
              sum->template mutable_data<T>(),
              &context_);
        }
        return true;
        */
    }
}

///---------------------------
pub struct MaxReductionOp<T,Context,const ROWWISE: bool> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

/**
  | Compute column-wise max reduction
  | of the input tensor.
  | 
  | This op takes one input, $X$, of shape
  | $BxMxN$, where $B$ is the batch size,
  | $M$ is number of rows, and $N$ is number
  | of columns.
  | 
  | The output of this op, $Y$, is a matrix
  | of shape $BxN$, with one row for each
  | element of the batch, and the same number
  | of columns as the input tensor.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc
  |
  */
register_cpu_operator!{ColwiseMax, MaxReductionOp<f32, CPUContext, false>}

num_inputs!{ColwiseMax, 1}

num_outputs!{ColwiseMax, 1}

inputs!{ColwiseMax, 
    0 => ("X", "A tensor of dimensions $B x M x N$ to compute columnwise-max. 
        Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of 
        each element of the batch, respectively.")
}

outputs!{ColwiseMax, 
    0 => ("Y", "The output tensor of shape $B x N$, where each row represents the 
        column-wise maximums for that element of the input batch.")
}

tensor_inference_function!{ColwiseMax, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
      vector<int64_t> output_dims = {in[0].dims()[0], in[0].dims()[2]};
      return vector<TensorShape>{
          CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};

        */
    }
}

#[test] fn colwise_max_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ColwiseMax",
        ["X"],
        ["Y"]
    )

    // Create X, simulating a batch of 2, 4x4 matricies
    X = np.random.randint(0,high=20,size=(2,4,4))
    print("X:\n",X)

    // Feed X into workspace
    workspace.FeedBlob("X", X.astype(np.float32))

    // Run op
    workspace.RunOperatorOnce(op)

    // Collect Output
    print("Y:\n", workspace.FetchBlob("Y"))

    ```

    **Result**

    ```

    X:
     [[[17 15  2  6]
      [ 8 12  6  0]
      [ 6  9  7  3]
      [ 4 13 16 13]]

     [[ 0  3  4 12]
      [18  1 17 12]
      [ 7 17 13 14]
      [12 17  2  1]]]
    Y:
     [[17. 15. 16. 13.]
     [18. 17. 17. 14.]]
    */
}

/**
  | Compute row-wise max reduction of the
  | input tensor.
  | 
  | This op takes one input, $X$, of shape
  | $BxMxN$, where $B$ is the batch size,
  | $M$ is number of rows, and $N$ is number
  | of columns.
  | 
  | The output of this op, $Y$, is a matrix
  | of shape $BxM$, with one row for each
  | element of the batch, and the same number
  | of columns as the number of rows of the
  | input tensor.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc
  |
  */
register_cpu_operator!{RowwiseMax, MaxReductionOp<f32, CPUContext, true>}

num_inputs!{RowwiseMax, 1}

num_outputs!{RowwiseMax, 1}

inputs!{RowwiseMax, 
    0 => ("X", "A tensor of dimensions $B x M x N$ to compute rowwise-max. 
        Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of 
        each element of the batch, respectively.")
}

outputs!{RowwiseMax, 
    0 => ("Y", "The output tensor of shape $B x M$, where each row 
        represents the row-wise maximums for that element of the input batch.")
}

#[test] fn rowwise_max_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "RowwiseMax",
        ["X"],
        ["Y"]
    )

    // Create X, simulating a batch of 2, 4x4 matricies
    X = np.random.randint(0,high=20,size=(2,4,4))
    print("X:\n",X)

    // Feed X into workspace
    workspace.FeedBlob("X", X.astype(np.float32))

    // Run op
    workspace.RunOperatorOnce(op)

    // Collect Output
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[[ 5 12 10  1]
      [ 4 16  2 15]
      [ 5 11 12 15]
      [15  4 17 19]]

     [[16  5  5 13]
      [17  2  1 17]
      [18  3 19  5]
      [14 16 10 16]]]
    Y:
     [[12. 16. 15. 19.]
     [16. 17. 19. 16.]]
    */
}

///---------------------------
impl<T, Context, const ROWWISE: bool> MaxReductionOp<T,Context,ROWWISE> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
        CAFFE_ENFORCE_EQ(X.dim(), 3);

        const int batch_size = X.dim32(0);
        const int M = X.dim32(1);
        const int N = X.dim32(2);

        auto* Y = Output(0, {batch_size, ROWWISE ? M : N}, at::dtype<T>());

        if (ROWWISE) {
          math::RowwiseMax<T, Context>(
              batch_size * M,
              N,
              X.template data<T>(),
              Y->template mutable_data<T>(),
              &context_);
        } else {
          const int input_size = N * M;
          for (int i = 0; i < batch_size; ++i) {
            math::ColwiseMax<T, Context>(
                M,
                N,
                X.template data<T>() + i * input_size,
                Y->template mutable_data<T>() + i * N,
                &context_);
          }
        }
        return true;
        */
    }
}

///--------------------
pub struct MaxReductionGradientOp<T,Context,const ROWWISE: bool> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

register_cpu_operator!{SumElements,           SumElementsOp<float, CPUContext>}
register_cpu_operator!{SumElementsInt,        SumElementsIntOp<int, CPUContext>}
register_cpu_operator!{SumSqrElements,        SumSqrElementsOp<CPUContext>}
register_cpu_operator!{SumElementsGradient,   SumElementsGradientOp<float, CPUContext>}
register_cpu_operator!{RowwiseMaxGradient,    MaxReductionGradientOp<float, CPUContext, true>}
register_cpu_operator!{ColwiseMaxGradient,    MaxReductionGradientOp<float, CPUContext, false>}

///------------------------------
pub struct GetSumElementsGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSumElementsGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SumElementsGradient",
            "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{SumElements, GetSumElementsGradient}

///------------------------------
num_inputs!{RowwiseMaxGradient, 3}

num_outputs!{RowwiseMaxGradient, 1}

pub struct GetRowwiseMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRowwiseMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RowwiseMaxGradient",
            "",
            vector<string>{I(0), O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{RowwiseMax, GetRowwiseMaxGradient}

///-------------------------------------
num_inputs!{ColumnMaxGradient, 3}

num_outputs!{ColumnMaxGradient, 1}

pub struct GetColwiseMaxGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetColwiseMaxGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ColwiseMaxGradient",
            "",
            vector<string>{I(0), O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{ColwiseMax, GetColwiseMaxGradient}

impl<T, Context> SumElementsGradientOp<T, Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // TODO: T21635077 fix float-divide-by-zero undefined behavior
      auto& X = Input(0);
      Tensor sum_grad(Input(1), CPU);

      auto* dX = Output(0, X.sizes(), at::dtype<T>());
      DCHECK_EQ(sum_grad.numel(), 1);
      math::Set<T, Context>(
          dX->numel(),
          static_cast<T>(
              sum_grad.template data<T>()[0] * (average_ ? 1.0 / X.numel() : 1)),
          dX->template mutable_data<T>(),
          &context_);
      return true;
        */
    }
}

impl<T, Context, const ROWWISE: bool> 
MaxReductionGradientOp<T, Context, ROWWISE> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto& Y = Input(1);
      auto& dY = Input(2);

      auto* dX = Output(0, X.sizes(), at::dtype<T>());

      CAFFE_ENFORCE_EQ(X.dim(), 3);

      const int batch_size = X.dim32(0);
      const int M = X.dim32(1);
      const int N = X.dim32(2);

      const T* Xdata = X.template data<T>();
      const T* Ydata = Y.template data<T>();
      const T* dYdata = dY.template data<T>();
      T* dXdata = dX->template mutable_data<T>();

      const int input_size = M * N;
      for (int i = 0; i < batch_size; ++i) {
        const T* Xdata_i = Xdata + i * input_size;
        T* dXdata_i = dXdata + i * input_size;
        if (ROWWISE) {
          const T* Ydata_i = Ydata + i * M;
          const T* dYdata_i = dYdata + i * M;
          for (int m = 0; m < M; ++m) {
            const T* Xdata_m = Xdata_i + m * N;
            T* dXdata_m = dXdata_i + m * N;
            for (int n = 0; n < N; ++n) {
              if (Xdata_m[n] == Ydata_i[m]) {
                dXdata_m[n] = dYdata_i[m];
              } else {
                dXdata_m[n] = static_cast<T>(0);
              }
            }
          }
        } else {
          const T* Ydata_i = Ydata + i * N;
          const T* dYdata_i = dYdata + i * N;
          for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
              const T* Xdata_m = Xdata_i + m * N;
              T* dXdata_m = dXdata_i + m * N;
              if (Xdata_m[n] == Ydata_i[n]) {
                dXdata_m[n] = dYdata_i[n];
              } else {
                dXdata_m[n] = static_cast<T>(0);
              }
            }
          }
        }
      }

      return true;
        */
    }
}
