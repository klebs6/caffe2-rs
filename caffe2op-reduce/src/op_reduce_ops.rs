crate::ix!();

use crate::{
    CPUContext,
    TensorShape,
    OperatorDef,
    GradientMakerBase,
    OperatorStorage,
};

pub struct ReduceOp<InputTypes,Context,Reducer> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,

    axes:       Vec<i32>,
    keep_dims:  i32, // {};
    reducer:    Reducer,
    phantomIT: PhantomData<InputTypes>,
}

impl<InputTypes,Context,Reducer> ReduceOp<InputTypes,Context,Reducer> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(this->template GetRepeatedArgument<int>("axes")),
            OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& X = Input(0);
        const int ndim = X.dim();
        const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
        if (axes_.empty()) {
          axes_.resize(ndim);
          std::iota(axes_.begin(), axes_.end(), 0);
        } else {
          for (auto& axis : axes_) {
            axis = X.canonical_axis_index(axis);
          }
          std::sort(axes_.begin(), axes_.end());
          CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
          CAFFE_ENFORCE_LT(
              axes_.back(),
              ndim,
              "Axes ids must be smaller than the dimensions of input.");
        }
        std::vector<int64_t> output_dims;
        output_dims.reserve(ndim);
        std::size_t cur_axis = 0;
        for (int i = 0; i < ndim; ++i) {
          if (cur_axis < axes_.size() && i == axes_[cur_axis]) {
            if (keep_dims_) {
              output_dims.push_back(1);
            }
            ++cur_axis;
          } else {
            output_dims.push_back(X_dims[i]);
          }
        }
        auto* Y = Output(0, output_dims, at::dtype<T>());

        std::vector<int> Y_dims = X_dims;
        for (const int axis : axes_) {
          Y_dims[axis] = 1;
        }

        return reducer_.template Forward<T>(
            X_dims,
            Y_dims,
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        */
    }
}


///---------------------------

pub struct ReduceGradientOp<InputTypes,Context,Reducer> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,

    axes:     Vec<i32>,// {};
    reducer:  Reducer,
    phantomIT: PhantomData<InputTypes>,
}

impl<InputTypes,Context,Reducer> ReduceGradientOp<InputTypes,Context,Reducer> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(this->template GetRepeatedArgument<int>("axes"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& dY = Input(0);
        const auto& X = Input(1);
        const auto& Y = Input(2);

        const int ndim = X.dim();
        if (axes_.empty()) {
          axes_.resize(ndim);
          std::iota(axes_.begin(), axes_.end(), 0);
        } else {
          for (auto& axis : axes_) {
            axis = X.canonical_axis_index(axis);
          }
          std::sort(axes_.begin(), axes_.end());
          CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
          CAFFE_ENFORCE_LT(
              axes_.back(),
              ndim,
              "Axes ids must be smaller than the dimensions of input.");
        }
        const std::vector<int> dX_dims(X.sizes().cbegin(), X.sizes().cend());
        std::vector<int> dY_dims = dX_dims;
        for (const int axis : axes_) {
          dY_dims[axis] = 1;
        }
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        return reducer_.template Backward<T>(
            dY_dims,
            dX_dims,
            dY.template data<T>(),
            X.template data<T>(),
            Y.template data<T>(),
            dX->template mutable_data<T>(),
            &context_);
        */
    }
}


pub struct MinReducer<Context> { 

    phantom: PhantomData<Context>,
}

impl<Context> MinReducer<Context> {
    
    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::ReduceMin<T, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            T(1),
            X_data,
            Y_data,
            context);
        return true;
        */
    }
    
    #[inline] pub fn backward<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}

///----------------------
pub struct MaxReducer<Context> {
    
    phantom: PhantomData<Context>,
}

impl<Context> MaxReducer<Context> {

    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::ReduceMax<T, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            T(1),
            X_data,
            Y_data,
            context);
        return true;
        */
    }
    
    #[inline] pub fn backward<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
        
        */
    }
}

///-----------------------------
pub struct SumReducer<Context> { 
    phantom: PhantomData<Context>,
}

impl<Context> SumReducer<Context> {

    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::ReduceSum<T, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            T(1),
            X_data,
            Y_data,
            context);
        return true;
        */
    }

    #[inline] pub fn backward<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Broadcast(
            dY_dims.size(),
            dY_dims.data(),
            dX_dims.size(),
            dX_dims.data(),
            T(1),
            dY_data,
            dX_data,
            context);
        return true;
        */
    }
}

///-----------------------------------
pub struct MeanReducer<Context> {
    
    phantom: PhantomData<Context>,
}

impl<Context> MeanReducer<Context> {

    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::ReduceMean<T, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            T(1),
            X_data,
            Y_data,
            context);
        return true;
        */
    }
    
    #[inline] pub fn backward<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            const int dY_size = std::accumulate(
            dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
        const int dX_size = std::accumulate(
            dX_dims.cbegin(), dX_dims.cend(), 1, std::multiplies<int>());
        math::Broadcast(
            dY_dims.size(),
            dY_dims.data(),
            dX_dims.size(),
            dX_dims.data(),
            static_cast<T>(dY_size) / static_cast<T>(dX_size),
            dY_data,
            dX_data,
            context);
        return true;
        */
    }
}

///---------------------------------------
pub struct L1Reducer<Context> {
    
    phantom: PhantomData<Context>,
}

impl<Context> L1Reducer<Context> {
    
    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::ReduceL1<T, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            T(1),
            X_data,
            Y_data,
            context);
        return true;
        */
    }
    
    #[inline] pub fn backward<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}

///---------------------------
pub struct L2Reducer<Context> {
    
    phantom: PhantomData<Context>,
}

impl<Context> L2Reducer<Context> {

    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
            math::ReduceL2<T, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            T(1),
            X_data,
            Y_data,
            context);
        return true;
        */
    }
    
    #[inline] pub fn backward<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}

#[inline] pub fn compute_reduce_min_max_gradient<T>(
    dY_dims: &Vec<i32>,
    dX_dims: &Vec<i32>,
    dY_data: *const T,
    x_data:  *const T,
    y_data:  *const T,
    dX_data: *mut T)  {

    todo!();
    /*
        const auto dX_size = c10::multiply_integers(dX_dims.cbegin(), dX_dims.cend());
      const int ndim = dX_dims.size();
      std::vector<int> index(ndim, 0);
      for (int dX_index = 0; dX_index < dX_size; ++dX_index) {
        const int dY_index =
            math::utils::GetIndexFromDims(ndim, dY_dims.data(), index.data());
        dX_data[dX_index] =
            Y_data[dY_index] == X_data[dX_index] ? dY_data[dY_index] : T(0);
        math::utils::IncreaseIndexInDims(ndim, dX_dims.data(), index.data());
      }
    */
}

#[inline] pub fn reduce_shape_inference(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    
    todo!();
    /*
        if (in.size() != 1) {
        return std::vector<TensorShape>{
            CreateTensorShape({}, TensorProto_DataType_UNDEFINED)};
      }

      const auto& dims = in.front().dims();
      ArgumentHelper helper(def);
      std::vector<TensorShape> out;
      out.emplace_back();
      auto& ts = out.back();
      auto axis = helper.GetRepeatedArgument<int32_t>("axes");
      std::sort(axis.begin(), axis.end());
      auto keepdims = helper.GetSingleArgument<bool>("keepdims", true);
      size_t cursor = 0;
      int32_t id = 0;
      for (const auto d : dims) {
        if (cursor < axis.size() && id == axis[cursor]) {
          if (keepdims) {
            ts.add_dims(d == 0 ? 0 : 1);
          }
          ++cursor;
        } else {
          ts.add_dims(d);
        }
        ++id;
      }
      if (ts.dims_size() == 0 && dims.size() != 0) {
        ts.add_dims(1);
      }
      if (cursor != axis.size()) {
        ts.set_unknown_shape(true);
      }
      ts.set_data_type(in.front().data_type());
      return out;
    */
}

impl MinReducer<CPUContext> {

    #[inline] pub fn backward_cpu<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            ComputeReduceMinMaxGradient(
          dY_dims, dX_dims, dY_data, X_data, Y_data, dX_data);
      return true;
        */
    }
}

impl MaxReducer<CPUContext> {

    #[inline] pub fn backward_cpu<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            ComputeReduceMinMaxGradient(
          dY_dims, dX_dims, dY_data, X_data, Y_data, dX_data);
      return true;
        */
    }
}

/**
  | Computes the min of the input tensor's
  | element along the provided axes.
  | 
  | The resulted tensor has the same rank
  | as the input if keepdims equal True.
  | 
  | If keepdims equal false, then the resulted
  | tensor have the reduced dimension pruned.
  |
  */
register_cpu_operator!{ReduceMin,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MinReducer<CPUContext>>
}

num_inputs!{ReduceMin, 1}

num_outputs!{ReduceMin, 1}

inputs!{ReduceMin, 
    0 => ("data", "An input tensor.")
}

outputs!{ReduceMin, 
    0 => ("reduced", "Reduced output tensor.")
}

args!{ReduceMin, 
    0 => ("axes", "A list of integers, along which to reduce."),
    1 => ("keepdims", "Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).")
}

///------------------------
register_cpu_operator!{ReduceMinGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MinReducer<CPUContext>>
}

num_inputs!{ReduceMinGradient, 3}

num_outputs!{ReduceMinGradient, 1}

/**
  | Computes the max of the input tensor's
  | element along the provided axes.
  | 
  | The resulted tensor has the same rank
  | as the input if keepdims equal True.
  | 
  | If keepdims equal false, then the resulted
  | tensor have the reduced dimension pruned.
  |
  */
register_cpu_operator!{
    ReduceMax,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MaxReducer<CPUContext>>
}

num_inputs!{ReduceMax, 1}

num_outputs!{ReduceMax, 1}

inputs!{ReduceMax, 
    0 => ("data", "An input tensor.")
}

outputs!{ReduceMax, 
    0 => ("reduced", "Reduced output tensor.")
}

args!{ReduceMax, 
    0 => ("axes", "A list of integers, along which to reduce."),
    1 => ("keepdims", "Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).")
}

///------------------------
register_cpu_operator!{
    ReduceMaxGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MaxReducer<CPUContext>>
}

num_inputs!{ReduceMaxGradient, 3}

num_outputs!{ReduceMaxGradient, 1}

///------------------------

#[test] fn reduce_sum_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceSum",
        ["X"],
        ["Y"],
        axes=(0,1),
        keepdims=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[5. 3. 7. 9. 5.]
       [4. 5. 1. 8. 3.]
       [1. 0. 9. 7. 6.]
       [7. 5. 0. 3. 1.]
       [6. 4. 4. 8. 3.]]

      [[8. 9. 6. 7. 7.]
       [5. 5. 4. 7. 0.]
       [9. 7. 6. 6. 7.]
       [7. 5. 2. 4. 2.]
       [4. 5. 1. 9. 4.]]]]
    Y:
    [[13. 12. 13. 16. 12.]
     [ 9. 10.  5. 15.  3.]
     [10.  7. 15. 13. 13.]
     [14. 10.  2.  7.  3.]
     [10.  9.  5. 17.  7.]]

    */
}

/**
  | Computes the **sum** of the input tensor's
  | elements along the provided `axes`.
  | The resulting tensor has the same rank
  | as the input if the `keepdims` argument
  | equals 1 (default). If `keepdims` is
  | set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
register_cpu_operator!{ReduceSum,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        SumReducer<CPUContext>>
}

num_inputs!{ReduceSum, 1}

num_outputs!{ReduceSum, 1}

inputs!{ReduceSum, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceSum, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceSum, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
}

tensor_inference_function!{ReduceSum, ReduceShapeInference}

///-----------------------------------
register_cpu_operator!{ReduceSumGradient,
    ReduceGradientOp<
        TensorTypes<i32, i64, f32, f64>,
        CPUContext,
        SumReducer<CPUContext>>
}

num_inputs!{ReduceSumGradient, 3}

num_outputs!{ReduceSumGradient, 1}

/**
  | Computes the **mean** of the input tensor's
  | elements along the provided `axes`.
  |
  | The resulting tensor has the same rank
  | as the input if the `keepdims` argument
  | equals 1 (default). 
  |
  | If `keepdims` is
  | set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
#[test] fn reduce_mean_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceMean",
        ["X"],
        ["Y"],
        axes=(0,1),
        keepdims=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[9. 0. 3. 6. 0.]
       [3. 4. 5. 0. 9.]
       [6. 9. 1. 1. 5.]
       [6. 2. 3. 7. 7.]
       [3. 1. 1. 0. 1.]]

      [[4. 3. 9. 8. 1.]
       [8. 2. 0. 4. 0.]
       [8. 9. 9. 0. 2.]
       [7. 2. 5. 8. 9.]
       [5. 9. 1. 9. 0.]]]]
    Y:
    [[6.5 1.5 6.  7.  0.5]
     [5.5 3.  2.5 2.  4.5]
     [7.  9.  5.  0.5 3.5]
     [6.5 2.  4.  7.5 8. ]
     [4.  5.  1.  4.5 0.5]]

    */
}

register_cpu_operator!{
    ReduceMean,
    ReduceOp<TensorTypes<float>, CPUContext, MeanReducer<CPUContext>>
}

num_inputs!{ReduceMean, 1}

num_outputs!{ReduceMean, 1}

inputs!{ReduceMean, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceMean, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{
    ReduceMean, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
}

tensor_inference_function!{
    ReduceMean, 
    ReduceShapeInference
}

///-----------------------------------
register_cpu_operator!{ReduceMeanGradient,
    ReduceGradientOp<TensorTypes<float>, CPUContext, MeanReducer<CPUContext>>
}

num_inputs!{ReduceMeanGradient, 3}

num_outputs!{ReduceMeanGradient, 1}

///-----------------------------------
impl L1Reducer<CPUContext> {

    #[inline] pub fn backward_cpu<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const float kEps = 1e-12f;
      const auto dX_size = c10::multiply_integers(dX_dims.cbegin(), dX_dims.cend());
      const int ndim = dX_dims.size();
      std::vector<int> index(ndim, 0);
      for (int dX_index = 0; dX_index < dX_size; ++dX_index) {
        const int dY_index =
            math::utils::GetIndexFromDims(ndim, dY_dims.data(), index.data());
        float temp = X_data[dX_index];
        if (temp < -kEps) {
          dX_data[dX_index] = -dY_data[dY_index];
        } else if (temp > kEps) {
          dX_data[dX_index] = dY_data[dY_index];
        } else {
          dX_data[dX_index] = T(0);
        }
        math::utils::IncreaseIndexInDims(ndim, dX_dims.data(), index.data());
      }
      return true;
        */
    }
}

impl L2Reducer<CPUContext> {

    #[inline] pub fn backward_cpu<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const float kEps = 1e-12f;
      const auto dX_size = c10::multiply_integers(dX_dims.cbegin(), dX_dims.cend());
      const int ndim = dX_dims.size();
      std::vector<int> index(ndim, 0);
      for (int dX_index = 0; dX_index < dX_size; ++dX_index) {
        const int dY_index =
            math::utils::GetIndexFromDims(ndim, dY_dims.data(), index.data());
        T norm = Y_data[dY_index];
        if (norm < kEps) {
          dX_data[dX_index] = dY_data[dY_index];
        } else {
          dX_data[dX_index] = dY_data[dY_index] * X_data[dX_index] / norm;
        }
        math::utils::IncreaseIndexInDims(ndim, dX_dims.data(), index.data());
      }
      return true;
        */
    }
}

/**
  | Computes the **L1 norm** of the input
  | tensor's elements along the provided
  | `axes`. The resulting tensor has the
  | same rank as the input if the `keepdims`
  | argument equals 1 (default). If `keepdims`
  | is set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
register_cpu_operator!{
    ReduceL1,
    ReduceOp<TensorTypes<float>, CPUContext, L1Reducer<CPUContext>>
}

num_inputs!{ReduceL1, 1}

num_outputs!{ReduceL1, 1}

inputs!{ReduceL1, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceL1, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceL1, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) 
        (default=1), else set to 0 to not keep the reduced dimension(s)")
}

#[test] fn reduce_l1_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceL1",
        ["X"],
        ["Y"],
        axes=(0,1),
        keepdims=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    ```

    **Result**

    ```

    X:
    [[[[ 2.  7.  6.  4.  5.]
       [ 2.  1.  9.  8.  7.]
       [ 4.  9.  1.  0.  0.]
       [ 6.  4.  0.  8.  1.]
       [ 1.  7.  1.  0.  2.]]

      [[ 5.  8.  1.  7.  7.]
       [ 4.  5.  6.  5.  4.]
       [ 1.  9.  6.  6.  3.]
       [ 6.  6.  8.  8.  4.]
       [ 2.  3.  5.  8.  1.]]]]

    Y:
    [[  7.  15.   7.  11.  12.]
     [  6.   6.  15.  13.  11.]
     [  5.  18.   7.   6.   3.]
     [ 12.  10.   8.  16.   5.]
     [  3.  10.   6.   8.   3.]]
    */
}

register_cpu_operator!{
    ReduceL1Gradient,
    ReduceGradientOp<TensorTypes<float>, CPUContext, L1Reducer<CPUContext>>
}

num_inputs!{ReduceL1Gradient, 3}

num_outputs!{ReduceL1Gradient, 1}

/**
  | Computes the **L2 norm** of the input
  | tensor's elements along the provided
  | `axes`.
  | 
  | The resulting tensor has the same rank
  | as the input if the `keepdims` argument
  | equals 1 (default). If `keepdims` is
  | set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
register_cpu_operator!{
    ReduceL2,
    ReduceOp<TensorTypes<float>, CPUContext, L2Reducer<CPUContext>>
}

num_inputs!{ReduceL2, 1}

num_outputs!{ReduceL2, 1}

inputs!{ReduceL2, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceL2, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceL2, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
}

inherit_onnx_schema!{ReduceL2, "ReduceMean"}

#[test] fn reduce_l2_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceL2",
        ["X"],
        ["Y"],
        axes=(0,1),
        keepdims=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[ 8.  0.  2.  5.  1.]
       [ 1.  3.  0.  4.  0.]
       [ 1.  3.  6.  7.  7.]
       [ 6.  9.  8.  4.  6.]
       [ 6.  1.  5.  7.  3.]]

      [[ 2.  4.  6.  2.  8.]
       [ 1.  1.  8.  0.  8.]
       [ 5.  9.  0.  3.  2.]
       [ 1.  7.  3.  7.  3.]
       [ 6.  8.  9.  8.  7.]]]]

    Y:
    [[  8.24621105   4.           6.3245554    5.38516474   8.06225777]
     [  1.41421354   3.1622777    8.           4.           8.        ]
     [  5.09901953   9.48683262   6.           7.6157732    7.28010988]
     [  6.08276272  11.40175438   8.54400349   8.06225777   6.70820379]
     [  8.48528099   8.06225777  10.29563046  10.63014603   7.6157732 ]]

    */
}

///--------------------------------
register_cpu_operator!{
    ReduceL2Gradient,
    ReduceGradientOp<TensorTypes<float>, CPUContext, L2Reducer<CPUContext>>
}

num_inputs!{ReduceL2Gradient, 3}

num_outputs!{ReduceL2Gradient, 1}

///--------------------------------
pub struct GetReduceGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<string>{GO(0), I(0), O(0)},
            std::vector<string>{GI(0)});
        */
    }
}

register_gradient!{ReduceMin,  GetReduceGradient}
register_gradient!{ReduceMax,  GetReduceGradient}
register_gradient!{ReduceSum,  GetReduceGradient}
register_gradient!{ReduceMean, GetReduceGradient}
register_gradient!{ReduceL1,   GetReduceGradient}
register_gradient!{ReduceL2,   GetReduceGradient}
