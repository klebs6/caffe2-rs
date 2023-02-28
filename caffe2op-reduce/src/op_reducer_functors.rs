/*!
  | Range reducers: can leverage that input
  | segment is continuous and provide special
  | implementation
  |
  */

crate::ix!();

use crate::{
    OpSchema,
    TensorShape,
    OperatorDef,
    CPUContext,
    Tensor
};

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct SumRangeReducer<T, CPUContext> {
    
    phantom: PhantomData<T>,
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> SumRangeReducer<T, CPUContext> {
    
    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {

        todo!();
        /*
            // do we need to go through wrapper in math.h?
        EigenVectorMap<T> out_vec(out, block_size);
        out_vec = ConstEigenMatrixMap<T>(in, block_size, blocks).rowwise().sum();
        */
    }
}

///-----------------------
pub struct SumRangeReducerGradient<T,Context> {
    
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

impl<T,Context> SumRangeReducerGradient<T,Context> {

    #[inline] pub fn invoke(&mut self, 
        block_size:   i64,
        blocks:       i64,
        segment_grad: *const T,
        data_grad:    *mut T,
        data_in:      *const T,
        data_out:     *const T,
        context:      *mut Context)  {

        todo!();
        /*
            // do we have some op that does it smartly with minimum number of memcpy?
        for (int64_t i = 0; i < blocks; ++i) {
          context->template CopySameDevice<T>(
              block_size, segment_grad, data_grad + block_size * i);
        }
        */
    }
}

///------------------------
pub struct SumRangeReducerDef {
    
    /*
  template <typename T, class Context>
  using Reducer = SumRangeReducer<T, Context>;

  template <typename T, class Context>
  using ReducerGradient = SumRangeReducerGradient<T, Context>;

  static constexpr const char* name = "Sum";
  static constexpr const char* doc =
      "Summation is done element-wise across slices of the input tensor and "
      "doesn't change the shape of the individual blocks.";
    */
}

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct LogSumExpRangeReducer<T, CPUContext> {

    r:  T, // default = 1
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> LogSumExpRangeReducer<T, CPUContext> {

    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {

        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          T max_value = std::numeric_limits<T>::lowest();
          for (int i = 0; i < blocks; ++i) {
            max_value = std::max(max_value, in[i * block_size + j]);
          }
          T scaled_exp_sum = 0;
          for (int i = 0; i < blocks; ++i) {
            scaled_exp_sum += std::exp(in[i * block_size + j] - max_value);
          }
          *(out++) = std::log(scaled_exp_sum) + max_value;
        }
        */
    }
}

///-------------------------
pub struct LogSumExpRangeReducerGradient<T,Context> {
    
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

impl<T,Context> LogSumExpRangeReducerGradient<T,Context> {
    
    /*
     | const T* segment_grad, // GO
     | T* data_grad, // GI
     | const T* data_in, // I
     | const T* data_out, // O
     */
    #[inline] pub fn invoke(&mut self, 
        block_size:   i64,
        blocks:       i64,
        segment_grad: *const T,
        data_grad:    *mut T,
        data_in:      *const T,
        data_out:     *const T,
        context:      *mut Context)  {

        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          const T out_grad = *(segment_grad++);
          const T offset = *(data_out++);
          for (int i = 0; i < blocks; ++i) {
            auto idx = i * block_size + j;
            data_grad[idx] = out_grad * std::exp(data_in[idx] - offset);
          }
        }
        */
    }
}

///-------------------------------
pub struct LogSumExpRangeReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = LogSumExpRangeReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = LogSumExpRangeReducerGradient<T, Context>;

      static constexpr const char* name = "LogSumExp";

      static constexpr const char* doc =
          "LogSumExp computes the element-wise log of the sum of exponentials of "
          "input slices. Operation doesn't change the shape of individual blocks.";
    */
}

///-------------------------
pub struct LogMeanExpRangeReducer<T> {
    context: CPUContext,
    
    phantom: PhantomData<T>,
}

impl<T> LogMeanExpRangeReducer<T> {

    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {

        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          T max_value = std::numeric_limits<T>::lowest();
          for (int i = 0; i < blocks; ++i) {
            max_value = std::max(max_value, in[i * block_size + j]);
          }
          T scaled_exp_sum = 0;
          for (int i = 0; i < blocks; ++i) {
            scaled_exp_sum += std::exp(in[i * block_size + j] - max_value);
          }
          scaled_exp_sum /= blocks;
          *(out++) = std::log(scaled_exp_sum) + max_value;
        }
        */
    }
}

///--------------------------------
pub struct LogMeanExpRangeReducerGradient<T,Context> {
    
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

impl<T,Context> LogMeanExpRangeReducerGradient<T,Context> {
    
    /*
      const T* segment_grad, // GO
      T* data_grad, // GI
      const T* data_in, // I
      const T* data_out, // O
          */
    #[inline] pub fn invoke(&mut self, 
        block_size:   i64,
        blocks:       i64,
        segment_grad: *const T,
        data_grad:    *mut T,
        data_in:      *const T,
        data_out:     *const T,
        context:      *mut Context)  {

        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          const T out_grad = *(segment_grad++);
          const T offset = *(data_out++);
          for (int i = 0; i < blocks; ++i) {
            auto idx = i * block_size + j;
            data_grad[idx] = out_grad * std::exp(data_in[idx] - offset) / blocks;
          }
        }
        */
    }
}

///----------------------
pub struct LogMeanExpRangeReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = LogMeanExpRangeReducer<T, Context>;
      template <typename T, class Context>
      using ReducerGradient = LogMeanExpRangeReducerGradient<T, Context>;
      static constexpr const char* name = "LogMeanExp";
      static constexpr const char* doc =
          "LogMeanExp computes the element-wise log of the mean of exponentials of "
          "input slices. Operation doesn't change the shape of individual blocks.";
    */
}

///-----------------------
pub struct MeanRangeReducer<T> {
    context: CPUContext,
    
    phantom: PhantomData<T>,
}

impl<T> MeanRangeReducer<T> {
    
    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {

        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          T avg_value = 0;
          for (int i = 0; i < blocks; ++i) {
            avg_value += in[i * block_size + j] / blocks;
          }
          *(out++) = avg_value;
        }
        */
    }
}

///-----------------
pub struct MeanRangeReducerGradient<T,Context> {
    
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

impl<T,Context> MeanRangeReducerGradient<T,Context> {
    
    /*
      const T* segment_grad, // GO
      T* data_grad, // GI
      const T* /*data_in*/, // I
      const T* /*data_out*/, // O
      */
          #[inline] pub fn invoke(&mut self, 
              block_size:   i64,
              blocks:       i64,
              segment_grad: *const T,
              data_grad:    *mut T,
              data_in:      *const T,
              data_out:     *const T,
              context:      *mut Context)  {
        
        todo!();
        /*
            const auto in_grad = 1.0 / blocks;
        for (int j = 0; j < block_size; ++j) {
          const T out_grad = *(segment_grad++);
          for (int i = 0; i < blocks; ++i) {
            auto idx = i * block_size + j;
            data_grad[idx] = out_grad * in_grad;
          }
        }
        */
    }
}

///----------------------
pub struct MeanRangeReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = MeanRangeReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = MeanRangeReducerGradient<T, Context>;

      static constexpr const char* name = "Mean";
      static constexpr const char* doc =
          "Mean computation is done element-wise, so that each element of the "
          "output slice corresponds to the average value of the respective "
          "elements in the input slices. Operation doesn't change the shape of "
          "individual blocks.";
    */
}

///------------------------
pub struct MaxRangeReducer<T, CPUContext> {
    
    phantom: PhantomData<T>,
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> MaxRangeReducer<T, CPUContext> {
    
    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {
        
        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          T max_value = std::numeric_limits<T>::lowest();
          for (int i = 0; i < blocks; ++i) {
            max_value = std::max(max_value, in[i * block_size + j]);
          }
          *(out++) = max_value;
        }
        */
    }
}

///---------------------------------
pub struct MaxRangeReducerGradient<T,Context> {
    
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

impl<T,Context> MaxRangeReducerGradient<T,Context> {
    
    /*
       const T* segment_grad, // GO
       T* data_grad, // GI
       const T* data_in, // I
       const T* data_out, // O
       */
    #[inline] pub fn invoke(&mut self, 
        block_size:   i64,
        blocks:       i64,
        segment_grad: *const T,
        data_grad:    *mut T,
        data_in:      *const T,
        data_out:     *const T,
        context:      *mut Context)  {

        todo!();
        /*
            std::memset(
            static_cast<void*>(data_grad), 0, blocks * block_size * sizeof(T));
        for (int j = 0; j < block_size; ++j) {
          const T out_grad = *(segment_grad++);
          const T out = data_out[j];
          for (int i = 0; i < blocks; ++i) {
            auto idx = i * block_size + j;
            if (out == data_in[idx]) {
              data_grad[idx] = out_grad;
            }
          }
        }
        */
    }
}

/**
  | Max computation is done element-wise,
  | so that each element of the output slice
  | corresponds to the max value of the respective
  | elements in the input slices. Operation
  | doesn't change the shape of individual
  | blocks. This implementation imitates
  | torch nn.Max operator.
  | 
  | If the maximum value occurs more than
  | once, the operator will return the first
  | occurrence of value. When computing
  | the gradient using the backward propagation,
  | the gradient input corresponding to
  | the first occurrence of the maximum
  | value will be used.
  |
  */
pub struct MaxRangeReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = MaxRangeReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = MaxRangeReducerGradient<T, Context>;

      static constexpr const char* name = "Max";
    */
}

/**
  | Incremental reducers: consume elements
  | one by one
  |
  */

/**
  | Base implementation, everything can
  | be overwritten
  |
  */
pub struct BaseReducer {
    
}

impl Reducer for BaseReducer {
    const InputCount: isize = 1;
}

impl BaseReducer {

    #[inline] pub fn finish<const FixedSize: i32>(&mut self, meta: &BaseReducerMeta, context: *mut CPUContext)  {
    
        todo!();
        /*
        
        */
    }
}

///-------------------------
pub struct BaseReducerMeta {
    block_size:   i64,
    block_shape:  Vec<i64>,
    first_dim:    bool,
}

impl BaseReducerMeta {
    
    pub fn new(first: Option<bool>) -> Self {
    
        let first: bool = first.unwrap_or(true);

        todo!();
        /*
            : first_dim(first)
        */
    }
    
    #[inline] pub fn compute_meta(
        &mut self, 
        dims: &[i32], 
        skip_dims: usize)  
    {
        todo!();
        /*
            first_dim ? block_shape.assign(dims.begin() + skip_dims, dims.end())
                : block_shape.assign(dims.begin(), dims.end() - skip_dims);
            block_size = first_dim ? size_from_dim_(skip_dims, dims)
                : size_from_dim_(dims.size() - skip_dims, dims);
        */
    }
    
    #[inline] pub fn observe_input(
        &mut self, 
        input:     i32,
        value:     &Tensor,
        skip_dims: i32)  
    {
        todo!();
        /*
            DCHECK_EQ(0, input);
            auto dims = value.sizes();
            computeMeta(dims, skip_dims);
        */
    }
    
    #[inline] pub fn append_output_shape(
        &mut self, 
        output_shape: *mut Vec<i64>)  
    {
        todo!();
        /*
            output_shape->insert(
                output_shape->end(), block_shape.begin(), block_shape.end());
        */
    }
    
    #[inline] pub fn get_output_shape(
        &mut self, 
        input: &TensorShape, 
        skip_dims: i32) -> Vec<i64> 
    {
        todo!();
        /*
            vector<int64_t> dims(in.dims().begin(), in.dims().end());
            computeMeta(dims, skip_dims);
            return block_shape;
        */
    }
}

///-----------------------------
pub struct BaseReducerGradient {
    
}

impl BaseReducerGradient {

    /// which of the original inputs are required for gradient computation
    #[inline] pub fn original_inputs() -> [i32; 0] {
        todo!();
        /*
           return std::array<int, 0>();
        */
    }
    
    #[inline] pub fn compute_length() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] pub fn num_aux_inputs_with_grads(def: &OperatorDef) -> i32 {
        
        todo!();
        /*
            return 0;
        */
    }
    
    #[inline] pub fn requires_data_input(def: &OperatorDef) -> bool {
        
        todo!();
        /*
            return false;
        */
    }

    /// True if the backward op requires the output of the forward op.
    #[inline] pub fn requires_forward_output() -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}


pub struct BaseReducerGradientMeta {

    block_size:   i64,
    block_shape:  Vec<i64>,
    first_dim:    bool,
}

impl BaseReducerGradientMeta {
    
    pub fn new(
        out_grad:  &Tensor,
        skip_dims: i32,
        first_dim: Option<bool>) -> Self {

        let first_dim: bool = first_dim.unwrap_or(true);

        todo!();
        /*
            : first_dim(first_dim) 

                auto dims = out_grad.sizes();
                first_dim ? block_shape.assign(dims.begin() + skip_dims, dims.end())
                    : block_shape.assign(dims.begin(), dims.end() - skip_dims);
                block_size = first_dim
                    ? out_grad.size_from_dim(skip_dims)
                    : out_grad.size_from_dim(out_grad.dim() - skip_dims);
        */
    }

    /// optional grad to populate
    #[inline] pub fn observe_original_input(
        &mut self, 
        original_input: i32,
        value:          &Tensor,
        input_grad:     *mut Tensor,
        skip_dims:      i32)  {

        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn append_grad_shape(&mut self, output_shape: *mut Vec<i64>)  {
        
        todo!();
        /*
            output_shape->insert(
                output_shape->end(), block_shape.begin(), block_shape.end());
        */
    }
}

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct SumReducer<T> {

    base:    BaseReducer,
    context: CPUContext,

    current_size:  i32,
    out:           *mut T,
    /*
       using FixedDispatch = FixedValues<1>;
       */
}

impl<T> SumReducer<T> {
    
    pub fn new(
        meta:    &BaseReducerMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : current_size_(0), out_(out) 

        // add a wrapper in Context for it
        if (meta.first_dim) {
          memset(out, 0, sizeof(T) * meta.block_size);
        }
        */
    }
    
    #[inline] pub fn process<const FixedSize: i32>(&mut self, 
        meta:    &BaseReducerMeta,
        input:   *const T,
        offset:  i64,
        context: *mut CPUContext)  {

        todo!();
        /*
            if (meta.first_dim) {
          math::AxpyFixedSize<T, CPUContext, FixedSize>(
              meta.block_size, 1, in, out_, context);
        } else {
          math::Sum<T, CPUContext>(
              meta.block_size, in, out_ + current_size_++, context);
        }
        */
    }
}

///-------------------------------------
pub struct SumReducerGradient<T,Context> {
    base:   BaseReducerGradient,
    s_grad: *const T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
    phantom: PhantomData<Context>,
}

impl<T,Context> SumReducerGradient<T,Context> {

    pub fn new(
        meta:    &BaseReducerGradientMeta,
        s_grad:  *const T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : s_grad_(s_grad)
        */
    }
    
    #[inline] pub fn fill_grad<const FixedSize: i32>(&mut self, 
        meta:      &BaseReducerGradientMeta,
        data_grad: *mut T,
        offset:    i64,
        context:   *mut Context,
        length:    i32)  {
    
        todo!();
        /*
            if (FixedSize == 1) { // static if
          *data_grad = *s_grad_;
        } else if (meta.first_dim) {
          context->template CopySameDevice<T>(meta.block_size, s_grad_, data_grad);
        } else {
          math::Set<T, Context>(length, s_grad_[offset], data_grad, context);
        }
        */
    }
}

/**
  | Summation is done element-wise across
  | slices of the input tensor and doesn't
  | change the shape of the individual blocks.
  |
  */
pub struct SumReducerDef {

    /*
      template <typename T, class Context>
      using Reducer = SumReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = SumReducerGradient<T, Context>;

      static constexpr const char* name = "Sum";
    */
}

impl SumReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
        
        */
    }
}

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct WeightedSumReducer<T> {
    base:    BaseReducer,
    context: CPUContext,
    out:     *mut T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
}

pub trait Reducer {
    const InputCount: isize;

}

impl<T> Reducer for WeightedSumReducer<T> {

    const InputCount: isize = 2;
}

impl<T> WeightedSumReducer<T> {


    pub fn new(
        meta:    &BaseReducerGradientMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : out_(out) 

        // do we have a wrapper for it?
        memset(out, 0, sizeof(T) * meta.block_size);
        */
    }
    
    #[inline] pub fn process<const FixedSize: i32>(&mut self, 
        meta:    &BaseReducerGradientMeta,
        input:   *const T,
        offset:  i64,
        context: *mut CPUContext)  {
    
        todo!();
        /*
            CAFFE_ENFORCE(
            meta.first_dim,
            "WeightedSumReducer implemented only for "
            "front dimensions reduction");
        math::AxpyFixedSize<T, CPUContext, FixedSize>(
            meta.block_size, meta.scalars[offset], in, out_, context);
        */
    }
}

///-----------------------
pub struct WeightedSumReducerMeta<T> {
    base: BaseReducerMeta,
    scalars:    *const T,
    first_dim:  bool,
}

impl<T> WeightedSumReducerMeta<T> {
    
    pub fn new(first: Option<bool>) -> Self {
    
        let first: bool = first.unwrap_or(true);

        todo!();
        /*
            : first_dim(first)
        */
    }
    
    #[inline] pub fn observe_input(&mut self, 
        input:     i32,
        value:     &Tensor,
        skip_dims: i32)  {

        todo!();
        /*
            if (input == 1) {
                CAFFE_ENFORCE_EQ(
                    skip_dims, value.dim(), "SCALARS mustn't have extra dimensions");
                scalars = value.data<T>();
                return;
            }
            BaseReducer::Meta::observeInput(input, value, skip_dims);
        */
    }
}

///-------------------------------------
pub struct WeightedSumReducerGradient<T,Context> {
    base: BaseReducerGradient,

    s_grad:  *const T,

    /**
      | using FixedDispatch = FixedValues<1>;
      |
      */
    phantom: PhantomData<Context>,
}

impl<T,Context> WeightedSumReducerGradient<T,Context> {

    /**
      | which of the original inputs are required
      | for gradient computation
      |
      */
    #[inline] pub fn original_inputs() -> [i32; 1] {
        todo!();
        /*
           return {{1}};
           */

    }
    
    #[inline] pub fn num_aux_inputs_with_grads(def: &OperatorDef) -> i32 {
        todo!();
        /*
            return GetFlagArgument(def, "grad_on_weights");
        */
    }
    
    #[inline] pub fn requires_data_input(def: &OperatorDef) -> bool {
        
        todo!();
        /*
            return numAuxInputsWithGrads(def) > 0;
        */
    }
    
    pub fn new(
        meta:    &BaseReducerGradientMeta,
        s_grad:  *const T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : s_grad_(s_grad)
        */
    }
    
    #[inline] pub fn fill_grad<const FixedSize: i32>(&mut self, 
        meta:      &BaseReducerGradientMeta,
        data_grad: *mut T,
        offset:    i64,
        context:   *mut Context,
        length:    i32)  {

        todo!();
        /*
            math::ScaleFixedSize<T, CPUContext, FixedSize>(
            meta.block_size, meta.scalars[offset], s_grad_, data_grad, context);
        */
    }

    /**
      | Special version which is called with
      | the main input too, used only if additional
      | input grad is requested
      |
      */
    #[inline] pub fn fill_grad_with_main_input<const FixedSize: i32>(&mut self, 
        meta:      &BaseReducerGradientMeta,
        data:      *const T,
        data_grad: *mut T,
        offset:    i64,
        context:   *mut Context,
        length:    i32)  {

        todo!();
        /*
            math::ScaleFixedSize<T, CPUContext, FixedSize>(
            meta.block_size, meta.scalars[offset], s_grad_, data_grad, context);
        math::Dot(
            meta.block_size, s_grad_, data, meta.scalars_grad + offset, context);
        */
    }
}

///------------------------
struct WeightedSumReducerGradientMeta<T> {
    base:          BaseReducerGradientMeta,
    scalars:       *const T,
    scalars_grad:  *mut T,
}

impl<T> WeightedSumReducerGradientMeta<T> {
    
    /**
      | Tensor* input_grad, // optional grad
      | to populate
      |
      */
    #[inline] pub fn observe_original_input(&mut self, 
        original_input: i32,
        value:          &Tensor,
        input_grad:     *mut Tensor,
        skip_dims:      i32)  {

        todo!();
        /*
            CAFFE_ENFORCE_EQ(1, original_input);
            scalars = value.data<T>();
            if (input_grad) {
                input_grad->ResizeLike(value);
                scalars_grad = input_grad->template mutable_data<T>();
            }
        */
    }
}

/**
  | Input slices are first scaled by SCALARS
  | and then summed element-wise.
  | 
  | It doesn't change the shape of the individual
  | blocks.
  |
  */
pub struct WeightedSumReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = WeightedSumReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = WeightedSumReducerGradient<T, Context>;

      static constexpr const char* name = "WeightedSum";
    */
}

impl WeightedSumReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor for the summation");
            schema.Input(
                1,
                "SCALARS",
                "Scalar multipliers for the input slices. Must be a vector with the "
                "length matching the number of slices");
            schema.Arg(
                "grad_on_weights",
                "Produce also gradient for `weights`. For now it's only supported in "
                "`Lengths`-based operators");
        */
    }
}

///-------------------------------------
pub struct MeanReducer<T, CPUContext> {
    base:              BaseReducer,
    out:               *mut T,
    current_size:      i32,

    /**
      | using FixedDispatch = FixedValues<1>;
      |
      */
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> MeanReducer<T, CPUContext> {
    
    pub fn new(
        meta:    &BaseReducerMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : out_(out), current_size_(0) 

        if (meta.first_dim) {
          memset(out, 0, sizeof(T) * meta.block_size);
        }
        */
    }
    
    #[inline] pub fn process<const FixedSize: i32>(&mut self, 
        meta:    &BaseReducerMeta,
        input:   *const T,
        offset:  i64,
        context: *mut CPUContext)  {

        todo!();
        /*
            if (meta.first_dim) {
          math::AxpyFixedSize<T, CPUContext, FixedSize>(
              meta.block_size, 1, in, out_, context);
        } else {
          math::Sum<T, CPUContext>(
              meta.block_size, in, out_ + current_size_, context);
        }
        current_size_++;
        */
    }
    
    #[inline] pub fn finish<const FixedSize: i32>(
        &mut self, 
        meta: &BaseReducerMeta, 
        context: *mut CPUContext)  {
    
        todo!();
        /*
            if (meta.first_dim) {
          if (current_size_ > 0) {
            math::ScaleFixedSize<T, CPUContext, FixedSize>(
                meta.block_size, 1.0 / current_size_, out_, out_, context);
          }
        } else {
          math::ScaleFixedSize<T, CPUContext, FixedSize>(
              current_size_, 1.0 / meta.block_size, out_, out_, context);
        }
        */
    }
}

///-------------------------------------
pub struct MeanReducerGradient<T,Context> {
    base: BaseReducerGradient,

    s_grad:  *const T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
    phantom: PhantomData<Context>,
}

impl<T,Context> MeanReducerGradient<T,Context> {
    
    #[inline] pub fn compute_length() -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    pub fn new(
        meta:    &BaseReducerGradientMeta,
        s_grad:  *const T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : s_grad_(s_grad)
        */
    }
    
    #[inline] pub fn fill_grad<const FixedSize: i32>(&mut self, 
        meta:      &BaseReducerGradientMeta,
        data_grad: *mut T,
        offset:    i64,
        context:   *mut Context,
        length:    i32)  {
    
        todo!();
        /*
            CAFFE_ENFORCE_GT(length, 0, "Segment length must be > 0");
        if (meta.first_dim) {
          math::ScaleFixedSize<T, CPUContext, FixedSize>(
              meta.block_size, 1.0 / length, s_grad_, data_grad, context);
        } else {
          math::Set<T, CPUContext>(
              length, s_grad_[offset] * 1.0f / length, data_grad, context);
        }
        */
    }
}

/**
  | Mean computes the element-wise mean
  | of the input slices.
  | 
  | Operation doesn't change the shape
  | of the individual blocks.
  |
  */
pub struct MeanReducerDef {
    /*
      template <typename T, class Context>
      using Reducer = MeanReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = MeanReducerGradient<T, Context>;

      static constexpr const char* name = "Mean";
    */

}

impl MeanReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
        
        */
    }
}

///-------------------------------------
pub struct MaxReducer<T> {

    base:          BaseReducer,
    context:       CPUContext,

    out:           *mut T,
    current_size:  i32,

    /*
       using FixedDispatch = FixedValues<1>;
       */
}

impl<T> MaxReducer<T> {
    
    pub fn new(
        meta:    &BaseReducerMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {
    
        todo!();
        /*
            : out_(out), current_size_(0) 

        // add a wrapper in Context for it
        memset(out, 0, sizeof(T) * meta.block_size);
        */
    }
    
    #[inline] pub fn process<const FixedSize: i32>(&mut self, 
        meta:    &BaseReducerMeta,
        input:   *const T,
        offset:  i64,
        context: *mut CPUContext)  {

        todo!();
        /*
            CAFFE_ENFORCE(
            meta.first_dim,
            "MaxReducer implemented only for front dimensions reduction");
        if (current_size_ > 0) {
          EigenVectorMap<T> output_vec(out_, meta.block_size);
          output_vec =
              output_vec.cwiseMax(ConstEigenVectorMap<T>(in, meta.block_size));
        } else {
          memcpy(out_, in, sizeof(T) * meta.block_size);
        }
        ++current_size_;
        */
    }
}

///-------------------------------------
pub struct MaxReducerGradient<T,Context> {
    base: BaseReducerGradient,

    s_grad:  *const T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
    phantom: PhantomData<Context>,
}

impl<T,Context> MaxReducerGradient<T,Context> {

    #[inline] pub fn requires_data_input(def: &OperatorDef) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn requires_forward_output() -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    pub fn new(
        meta:    &BaseReducerGradientMeta,
        s_grad:  *const T,
        context: *mut CPUContext) -> Self {
    
        todo!();
        /*
            : s_grad_(s_grad)
        */
    }
    
    #[inline] pub fn fill_grad_with_main_input_and_forward_output<const FixedSize: i32>(&mut self, 
        meta:           &BaseReducerGradientMeta,
        data:           *const T,
        data_grad:      *mut T,
        forward_output: *const T,
        offset:         i64,
        context:        *mut Context,
        length:         i32)  {

        todo!();
        /*
            for (int64_t i = 0; i < meta.block_size; ++i) {
          data_grad[i] = data[i] == forward_output[i] ? s_grad_[i] : 0;
        }
        */
    }
}

/**
  | Max computes the element-wise max of
  | the input slices.
  | 
  | Operation doesn't change the shape
  | of the individual blocks.
  |
  */
pub struct MaxReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = MaxReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = MaxReducerGradient<T, Context>;

      static constexpr const char* name = "Max";
    */
}

impl MaxReducerDef {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
        
        */
    }
}
