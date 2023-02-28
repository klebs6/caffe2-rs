crate::ix!();

use crate::{
    GradientMakerBase,
    Reducer,
    OpSchemaCost,
    OperatorDef,
    OperatorStorage,
    Tensor,
    CPUContext,
    TensorShape,
    MaxReducerDef,
    OpSchema,
};

declare_export_caffe2_op_to_c10!{LengthsSum}

declare_export_caffe2_op_to_c10!{LengthsMean}

declare_export_caffe2_op_to_c10!{LengthsMax}

pub struct BaseInputAccessor<TData> {

    data:  *const c_void, // default = nullptr
    phantom: PhantomData<TData>,
}

impl<TData> BaseInputAccessor<TData> {
    
    #[inline] pub fn observe_input(&mut self, data_input: &Tensor) -> bool {
        
        todo!();
        /*
            data_ = dataInput.raw_data();
        return dataInput.template IsType<TData>();
        */
    }
    
    #[inline] pub fn get_block_ptr(&mut self, 
        in_block_size: i64,
        idx:           i64,
        blocks:        Option<i64>) -> *const TData {
        let blocks: i64 = blocks.unwrap_or(1);

        todo!();
        /*
            return static_cast<const TData*>(data_) + in_block_size * idx;
        */
    }
}

/**
  | Range reducer ops: leverage that input segment
  | is continuous and allow reducer functors to do
  | something special
  |
  | Note: for now there are no real use cases for
  | it yet :)
  |
  | Also, doesn't support additional arguments for
  | now
  */

/**
  | Base implementation for segment reduction
  | op that leverages continuity of the
  | data
  | 
  | Assumes that segments are sorted and
  | there are no skip indices class InputAccessor
  | = BaseInputAccessor<T>>
  |
  */
pub struct AbstractSortedSegmentRangeOp<T,SIndex,Context,RangeReducer,InputAccessor> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:             OperatorStorage,
    context:             Context,

    input_accessor:      InputAccessor,
    phantom:             PhantomData<T>,
    phantomSIndex:       PhantomData<SIndex>,
    phantomRangeReducer: PhantomData<RangeReducer>,
}

input_tags!{
    AbstractSortedSegmentRangeOp {
        Data,
        SegmentIds
    }
}

impl<T,SIndex,Context,RangeReducer,InputAccessor> 
AbstractSortedSegmentRangeOp<T,SIndex,Context,RangeReducer,InputAccessor> {

    const kNumInputs: i32 = 2;
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& dataInput = Input(DATA);
        auto& segment_ids = Input(SEGMENT_IDS);

        CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
        auto N = segment_ids.size(0);
        CAFFE_ENFORCE_EQ(
            N,
            dataInput.size(0),
            "SEGMENT_IDS must have the same length as outer dimension of DATA");

        OPERATOR_NEEDS_FEATURE(
            inputAccessor_.observeInput(dataInput),
            "Unsupported input type: ",
            dataInput.dtype().name(),
            ".");

        const SIndex* s_ids = segment_ids.template data<SIndex>();

        const SIndex K = N > 0 ? s_ids[N - 1] + 1 : 0;
        auto shape = dataInput.sizes().vec();
        shape[0] = K;
        auto* output = Output(0, shape, at::dtype<T>());

        T* out = output->template mutable_data<T>();

        if (N == 0) {
          return true;
        }

        int64_t block_size = dataInput.numel() / N;

        // Assume the segments are sorted and there are no gaps
        CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
        for (int64_t i = 0; i < N;) {
          int64_t start = i;
          for (++i; i < N && s_ids[start] == s_ids[i]; ++i)
            ;

          RangeReducer()(
              block_size,
              i - start,
              inputAccessor_.getBlockPtr(block_size, start, i - start),
              out + block_size * s_ids[start],
              &context_);

          // check correctness of the next segment
          if (i < N) {
            CAFFE_ENFORCE_EQ(
                s_ids[start] + 1,
                s_ids[i],
                "Indices must be sorted and not have gaps");
          }
        }
        return true;
        */
    }
}

///------------------------------
pub struct AbstractSortedSegmentRangeGradientOp<T,SIndex,Context,RangeReducerGradient> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    //USE_SIMPLE_CTOR_DTOR(AbstractSortedSegmentRangeGradientOp);
    storage:                     OperatorStorage,
    context:                     Context,
    phantom:                     PhantomData<T>,
    phantomSIndex:               PhantomData<SIndex>,
    phantomRangeReducerGradient: PhantomData<RangeReducerGradient>,
}

input_tags!{
    AbstractSortedSegmentRangeGradientOp {
        DataIn,
        DataOut,
        SegmentGrads,
        SegmentIds
    }
}

impl<T,SIndex,Context,RangeReducerGradient> 
AbstractSortedSegmentRangeGradientOp<T,SIndex,Context,RangeReducerGradient> {

    const kNumInputs: i32 = 4;

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // TODO(azzolini): avoid using input/output if not used by a particular op
        auto& data_in = Input(DATA_IN);
        auto& data_out = Input(DATA_OUT);
        auto& segment_grads = Input(SEGMENT_GRADS);
        auto& segment_ids = Input(SEGMENT_IDS);

        CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
        int64_t N = segment_ids.size(0);

        const SIndex* s_ids = segment_ids.template data<SIndex>();
        const T* s_grads = segment_grads.template data<T>();
        const T* d_in = data_in.template data<T>();
        const T* d_out = data_out.template data<T>();

        auto shape = segment_grads.sizes().vec();
        shape[0] = N;
        auto* data_grads = Output(0, shape, at::dtype<T>());

        const SIndex K = segment_grads.size(0);
        T* out = data_grads->template mutable_data<T>();

        if (N == 0) {
          return true;
        }

        int64_t block_size = segment_grads.size_from_dim(1);

        // Assume the segments are sorted and there are no gaps
        CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
        // repeat the check from forward op
        CAFFE_ENFORCE_EQ(
            K - 1, s_ids[N - 1], "Indices must be sorted and not have gaps");
        for (int64_t i = 0; i < N;) {
          int64_t start = i;
          for (++i; i < N && s_ids[start] == s_ids[i]; ++i)
            ;

          auto expanded_idx = block_size * start;
          auto reduced_idx = block_size * s_ids[start];
          RangeReducerGradient()(
              block_size,
              i - start,
              s_grads + reduced_idx,
              out + expanded_idx,
              d_in + expanded_idx,
              d_out + reduced_idx,
              &context_);

          // check correctness of the next segment
          if (i < N) {
            CAFFE_ENFORCE_EQ(
                s_ids[start] + 1,
                s_ids[i],
                "Indices must be sorted and not have gaps");
          }
        }
        return true;
        */
    }
}

/**
  | Applies '{op}' to each segment of input
  | tensor. In order to allow for more efficient
  | implementation of '{op}', the input
  | segments have to be contiguous and non-empty.
  | 
  | SEGMENT_IDS is a vector that maps each
  | of the first dimension slices of the
  | 
  | DATA to a particular group (segment).
  | Values belonging to the same segment
  | are aggregated together.
  | 
  | The first dimension of the output is
  | equal to the number of input segments,
  | i.e. `SEGMENT_IDS[-1]+1`. Other dimensions
  | are inherited from the input tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractSortedSegmentRangeDef<T,SIndex,Context,ReducerDef> {

    /*
    |    pub type OpDef = ReducerDef;
    |
    | using ForwardOp = AbstractSortedSegmentRangeOp<
    |   T,
    |   SIndex,
    |   Context,
    |   typename ReducerDef::template Reducer<T, Context>>;
    |
    | using BackwardOp = AbstractSortedSegmentRangeGradientOp<
    |   T,
    |   SIndex,
    |   Context,
    |   typename ReducerDef::template ReducerGradient<T, Context>>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSortedSegmentRangeDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor to be aggregated");
            schema.Input(
                1,
                "SEGMENT_IDS",
                "Vector with the same length as the first dimension of DATA "
                "and values in the range 0..K-1 and in increasing order that "
                "maps each slice of DATA to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated tensor with the first dimension of K and the "
                "other dimentsions inherited from DATA");
        */
    }
}

pub struct GetSortedSegmentRangeGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSortedSegmentRangeGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
              "SortedSegmentRange" + ReducerDef::name + "Gradient",
              "",
              vector<string>{I(0), O(0), GO(0), I(1)},
              // no gradient on segment_ids!
              vector<string>{GI(0)});
        */
    }
}

/**
  | Incremental reducer ops: assume that
  | reducer consumes pieces of data one
  | by one. Also, supports additional arguments
  | passed to reducer, e.g. scalers for
  | weighted sum.
  | 
  | -----------
  | @note
  | 
  | in current implementation additional
  | inputs are considered auxiliary constants
  | and have limitations:
  | 
  | - there is no gradient computation for
  | auxiliary inputs
  | 
  | - auxiliary inputs aren't affected
  | by fused embedding lookup in operations
  | like sparse_sorted_segment
  |
  */

/**
  | @brief
  | 
  | Simple non-segmented reduction over
  | the first few dimensions of the tensor
  | 
  | Inputs:
  | 
  | 0: DATA - input embedding to do lookups
  | in
  | 
  | 1..P: AUX_ARG_<I> - optional additional
  | arguments to be passed to the reducer
  | 
  | Args: num_reduce_dim (default 1) -
  | the number of dims in front of the tensor
  | to reduce
  | 
  | Output:
  | 
  | Tensor without the first `num_dim`
  | dimensions of DATA class InputAccessor
  | = BaseInputAccessor<T>>
  |
  */
pub struct AbstractReduceFrontOrBackOp<T,Context,Reducer,const FirstDim: bool,InputAccessor> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,

    num_reduce_dims:  i32,
    input_accessor:   InputAccessor,
    phantom: PhantomData<T>,
    phantomReducer: PhantomData<Reducer>,
}

impl<T,Context,R: Reducer,const FirstDim: bool,InputAccessor> 
AbstractReduceFrontOrBackOp<T,Context,R,FirstDim,InputAccessor> {

    const kNumInputs: isize = <R as Reducer>::InputCount;

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "num_reduce_dim", num_reduce_dims_, 1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(0);
        // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t in_block_size = FirstDim
            ? data.size_from_dim(num_reduce_dims_)
            : data.size_to_dim(data.dim() - num_reduce_dims_);
        return DispatchHelper<typename Reducer::FixedDispatch>::call(
            this, in_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& data = Input(0);

        CAFFE_ENFORCE_LE(num_reduce_dims_, data.dim());

        typename Reducer::Meta ctx(FirstDim);
        ctx.observeInput(0, data, num_reduce_dims_);
        for (int i = 1; i < <R as Reducer>::InputCount; ++i) {
          auto& aux_in = Input(i);
          ctx.observeInput(i, aux_in, num_reduce_dims_);
        }

        OPERATOR_NEEDS_FEATURE(
            inputAccessor_.observeInput(data),
            "Unsupported input type: ",
            data.dtype().name(),
            ".");

        vector<int64_t> shape;
        ctx.appendOutputShape(&shape);
        auto* output = Output(0, shape, at::dtype<T>());

        T* out = output->template mutable_data<T>();

        const int block_size = FirstDim
            ? data.size_from_dim(num_reduce_dims_)
            : data.size_from_dim(data.dim() - num_reduce_dims_);

        const int num_blocks = block_size > 0 ? data.numel() / block_size : 0;

        Reducer r(ctx, out, &context_);
        for (int64_t i = 0; i < num_blocks; ++i) {
          r.template process<FixedSize>(
              ctx, inputAccessor_.getBlockPtr(block_size, i), i, &context_);
        }
        r.template finish<FixedSize>(ctx, &context_);
        return true;
        */
    }
}

/**
  | bool FirstDim = true>
  |
  */
pub struct AbstractReduceFrontOrBackGradientOp<T,Context,ReducerGradient,const FirstDim: bool> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    num_reduce_dims:  i32,
    phantom: PhantomData<T>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

pub enum _InputTags {
    REDUCTION_GRAD,// = <ReducerGradient as HasOriginalInputs>::original_inputs_size(),
    SOURCE_SHAPE,
}

impl<T,Context,ReducerGradient: HasOriginalInputs,const FirstDim: bool> AbstractReduceFrontOrBackGradientOp<T,Context,ReducerGradient,FirstDim> {

    const kNumInputs: isize = todo!(); // <ReducerGradient as HasOriginalInputs>::original_inputs_size() + 2;
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "num_reduce_dim", num_reduce_dims_, 1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t grad_block_size = Input(REDUCTION_GRAD).numel();
        return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
            this, grad_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& reduction_grad = Input(REDUCTION_GRAD);
        auto& source_shape = this->template Input<Tensor>(SOURCE_SHAPE, CPU);

        typename ReducerGradient::Meta ctx(reduction_grad, 0, FirstDim);
        for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
          auto& aux_in = Input(i);
          ctx.observeOriginalInput(
              ReducerGradient::originalInputs()[i],
              aux_in,
              nullptr, /*no grad*/
              num_reduce_dims_);
        }

        const T* r_grad = reduction_grad.template data<T>();

        CAFFE_ENFORCE_LE(num_reduce_dims_, source_shape.numel());

        vector<int64_t> shape(
            source_shape.template data<int64_t>(),
            source_shape.template data<int64_t>() + source_shape.numel());

        auto* data_grads = Output(0, shape, at::dtype<T>());

        int64_t block_size = FirstDim
            ? data_grads->size_from_dim(num_reduce_dims_)
            : data_grads->size_from_dim(data_grads->dim() - num_reduce_dims_);
        int64_t block_num = block_size > 0 ? data_grads->numel() / block_size : 0;

        T* out = data_grads->template mutable_data<T>();

        ReducerGradient r(ctx, r_grad, &context_);
        for (int64_t i = 0; i < block_num; ++i) {
          r.template fillGrad<FixedSize>(
              ctx,
              out + block_size * i,
              i,
              &context_,
              FirstDim ? block_num : block_size);
        }
        return true;
        */
    }
}

/**
  | Reduces the input tensor along the first
  | dimension of the input tensor by applying
  | '{op}'. This op acts in a similar way
  | to SortedSegment{op} and
  | 
  | UnsortedSegment{op} but as if all input
  | slices belong to a single segment.
  | 
  | {op_doc}
  |
  */
pub struct AbstractReduceFrontDef<T,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "ReduceFront";
      |
      | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractReduceFrontOrBackOp<
      |     T,
      |     Context,
      |     typename ReducerDef::template Reducer<T, Context>,
      |     true>;
      |
      | using BackwardOp = AbstractReduceFrontOrBackGradientOp<T, Context, ReducerGradient, true>;
      |
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,Context,ReducerDef> AbstractReduceFrontDef<T,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(
                0, "DATA", "Input tensor to be reduced on the first dimension");
            schema.TensorInferenceFunction([](const OperatorDef& def,
                                              const vector<TensorShape>& in) {
              CAFFE_ENFORCE_EQ(1, in.size());
              ArgumentHelper helper(def);
              int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1);
              typename ReducerDef::template Reducer<T, Context>::Meta ctx(true);
              vector<int64_t> out_dims = ctx.getOutputShape(in[0], num_reduce_dims);
              return vector<TensorShape>{
                  CreateTensorShape(out_dims, in[0].data_type())};
            });
            ReducerDef::PopulateSchema(schema);
        */
    }

}

pub struct GetReduceFrontGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceFrontGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // Have utility function generating these names?
          string tmp_dims = "_" + O(0) + "_dims";

          vector<string> grad_ins;
          for (const int i : ReducerGradient::originalInputs()) {
            grad_ins.push_back(I(i));
          }
          grad_ins.push_back(GO(0));
          grad_ins.push_back(tmp_dims);

          vector<Argument> args;
          if (ArgumentHelper::HasArgument(def_, "num_reduce_dim")) {
            args.push_back(GetArgument(def_, "num_reduce_dim"));
          }
          // FIXME: pass in num_reduce_dims?!
          return vector<OperatorDef>{
              CreateOperatorDef(
                  "Shape", "", vector<string>{I(0)}, vector<string>{tmp_dims}),
              CreateOperatorDef(
                  string(basename) + ReducerDef::name + "Gradient",
                  "",
                  grad_ins,
                  // no gradient on auxiliary inputs for now
                  vector<string>{GI(0)}),
          };
        */
    }
}

/**
  | Reduces the input tensor along the last
  | dimension of the input tensor by applying
  | '{op}'. This op acts in a similar way
  | to SortedSegment{op} and
  | 
  | UnsortedSegment{op} but as if all input
  | slices belong to a single segment.
  | 
  | {op_doc}
  |
  */
pub struct AbstractReduceBackDef<T,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      | static constexpr const char* basename = "ReduceBack";
      |
      | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractReduceFrontOrBackOp<
      |     T,
      |     Context,
      |     typename ReducerDef::template Reducer<T, Context>,
      |     false>;
      |
      | using BackwardOp = AbstractReduceFrontOrBackGradientOp<T, Context, ReducerGradient, false>;
    */
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,Context,ReducerDef> AbstractReduceBackDef<T,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(
                0, "DATA", "Input tensor to be reduced on the first dimension");
            schema.TensorInferenceFunction([](const OperatorDef& def,
                                              const vector<TensorShape>& in) {
              CAFFE_ENFORCE_EQ(1, in.size());
              ArgumentHelper helper(def);
              int num_reduce_dims = helper.GetSingleArgument<int>("num_reduce_dim", 1);
              typename ReducerDef::template Reducer<T, Context>::Meta ctx(false);
              vector<int64_t> out_dims = ctx.getOutputShape(in[0], num_reduce_dims);
              return vector<TensorShape>{
                  CreateTensorShape(out_dims, in[0].data_type())};
            });
            ReducerDef::PopulateSchema(schema);
        */
    }
}

pub struct GetReduceBackGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReduceBackGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // Have utility function generating these names?
          string tmp_dims = "_" + O(0) + "_dims";

          vector<string> grad_ins;
          for (const int i : ReducerGradient::originalInputs()) {
            grad_ins.push_back(I(i));
          }
          grad_ins.push_back(GO(0));
          grad_ins.push_back(tmp_dims);

          vector<Argument> args;
          if (ArgumentHelper::HasArgument(def_, "num_reduce_dim")) {
            args.push_back(GetArgument(def_, "num_reduce_dim"));
          }
          // FIXME: pass in num_reduce_dims?!
          return vector<OperatorDef>{
              CreateOperatorDef(
                  "Shape", "", vector<string>{I(0)}, vector<string>{tmp_dims}),
              CreateOperatorDef(
                  string(basename) + ReducerDef::name + "Gradient",
                  "",
                  grad_ins,
                  // no gradient on auxiliary inputs for now
                  vector<string>{GI(0)}),
          };
        */
    }
}

/**
 | @brief Segment reduction op with optional fused
 | embedding lookup
 |
 | Base implementation for SortedSegmentXXX and
 | SparseSortedSegmentXXX depending on SparseFused
 |  static argument.
 |
 | Inputs:
 |   0: DATA - input embedding to do lookups in
 |   1..P: AUX_ARG_<I> - optional additional arguments to be passed to the
 |                       reducer, should have the same first dimension as
 |                       SEGMENT_IDS (e.g. scalars in WeightedSum)
 |   # if SparseFused == true:
 |   P+1: INDICES - 1-D vector with indices to look up in DATA. Should have the
 |                  same dimension as SEGMENT_IDS
 |   # P+1 if SparseFused == false:
 |   P+1 or P+2: SEGMENT_IDS - sorted segment ids 1-D vector
 |
 | Output:
 |
 |   Tensor with first dimension of K, where K is
 |   the max segment id + 1. Rest of dimensions are
 |    decided by reducer but usually are the same
 |    size as extra dimensions of DATA
 |
 |  bool SparseFused = true,
 |  class InputAccessor = BaseInputAccessor<T>>
 */
pub struct AbstractSortedSegmentOp<T,SIndex,Context,Reducer,const SparseFused: bool,InputAccessor> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage:         OperatorStorage,
    context:         Context,

    input_accessor:  InputAccessor,
    phantom:         PhantomData<T>,
    phantomSIndex:   PhantomData<SIndex>,
    phantomReducer:  PhantomData<Reducer>,
}

/**
  | TODO: figure out what the two comments
  | below break*, if anything
  |
  */
pub enum AbstractSortedSegmentOpInputTags {
    INDICES,    // = <R as Reducer>::InputCount,
    SEGMENT_IDS,// = <R as Reducer>::InputCount + ternary![SparseFused,1,0]
}

impl<T,SIndex,Context,R: Reducer,const SparseFused: bool,InputAccessor> 
AbstractSortedSegmentOp<T,SIndex,Context,R,SparseFused,InputAccessor> {

    const kSelfInputs: isize = ternary![SparseFused, 2, 1];
    const kNumInputs:  isize = <R as Reducer>::InputCount + Self::kSelfInputs;

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (SparseFused) {
          return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
              this, Input(INDICES));
        } else {
          // type doesn't matter
          return DoRunWithType<int64_t>();
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self) -> bool {
    
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t in_block_size = Input(0).size_from_dim(1);
        return DispatchHelper<typename Reducer::FixedDispatch, IndexType>::call(
            this, in_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<IndexType, const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& dataInput = Input(0);
        auto& segment_ids = Input(SEGMENT_IDS);

        CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
        int64_t N = segment_ids.size(0);
        const int64_t M = dataInput.size(0);

        const IndexType* idxs;
        if (SparseFused) { // static if
          auto& indices = Input(INDICES);
          CAFFE_ENFORCE_EQ(1, indices.dim(), "INDICES must be a vector");
          CAFFE_ENFORCE_EQ(
              N,
              indices.size(0),
              "SEGMENT_IDS must have the same length as INDICES");
          idxs = indices.template data<IndexType>();
        } else {
          CAFFE_ENFORCE_EQ(
              N, M, "DATA must have the same first dimension as SEGMENT_IDS");
        }

        // It would probably look nicer with varargs templates but it's too much
        // metaprogramming
        typename Reducer::Meta ctx;
        ctx.observeInput(0, dataInput, 1);
        for (int i = 1; i < <R as Reducer>::InputCount; ++i) {
          auto& aux_in = Input(i);
          CAFFE_ENFORCE_EQ(
              N,
              aux_in.size(0),
              "Input ",
              i,
              " must have the same first dim as SEGMENT_IDS");
          ctx.observeInput(i, aux_in, 1);
        }

        OPERATOR_NEEDS_FEATURE(
            inputAccessor_.observeInput(dataInput),
            "Unsupported input type: ",
            dataInput.dtype().name(),
            ".");

        const SIndex* s_ids = segment_ids.template data<SIndex>();

        const SIndex K = N > 0 ? s_ids[N - 1] + 1 : 0;
        vector<int64_t> shape;
        shape.push_back(K);
        ctx.appendOutputShape(&shape);
        auto* output = Output(0, shape, at::dtype<T>());

        T* out = output->template mutable_data<T>();
        if (N == 0) {
          return true;
        }
        int64_t in_block_size = dataInput.size_from_dim(1);
        int64_t out_block_size = output->size_from_dim(1);

        // Assume the segments are sorted and there are no gaps
        CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
        for (int64_t i = 0; i < N;) {
          int64_t start = i;

          Reducer r(ctx, out + out_block_size * s_ids[start], &context_);
          for (; i < N && s_ids[start] == s_ids[i]; ++i) {
            IndexType idx;
            if (SparseFused) { // static if
              CAFFE_ENFORCE(
                  0 <= idxs[i] && idxs[i] < M,
                  "Index out of bounds: ",
                  idxs[i],
                  ", range 0 to ",
                  M);
              idx = idxs[i];
            } else {
              idx = i;
            }
            r.template process<FixedSize>(
                ctx, inputAccessor_.getBlockPtr(in_block_size, idx), i, &context_);
          }

          r.template finish<FixedSize>(ctx, &context_);
          // check correctness of the next segment
          if (i < N) {
            CAFFE_ENFORCE_EQ(
                s_ids[start] + 1,
                s_ids[i],
                "Indices must be sorted and not have gaps");
          }
        }
        return true;
        */
    }
}

/**
  | Gradient actually doesn't depend on
  | whether sparse lookup is fused or not
  |
  */
pub struct AbstractSortedSegmentGradientOp<T,SIndex,Context,ReducerGradient: HasOriginalInputs> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:                OperatorStorage,
    context:                Context,
    phantom:                PhantomData<T>,
    phantomSIndex:          PhantomData<SIndex>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

pub trait HasOriginalInputs {
    fn original_inputs_size() -> isize;
}

//TODO: does the below comment break something?
pub enum AbstractSortedSegmentGradientOpInputTags {
    SEGMENT_GRADS,// = <ReducerGradient as  HasOriginalInputs>::original_inputs_size(),
    SEGMENT_IDS
}

impl<T,SIndex,Context,ReducerGradient: HasOriginalInputs> AbstractSortedSegmentGradientOp<T,SIndex,Context,ReducerGradient> {

    /**
      | Input layout:
      |
      |   orig_arg1, orig_arg2, ..., orig_argN,
      |   SEGMENT_GRADS, SEGMENT_IDS
      |
      | orig_argXs represent original op's inputs
      | and will be passed to the reducer directly
      */
    const kNumInputs: isize = todo!(); // <ReducerGradient as HasOriginalInputs>::original_inputs_size() + 2;

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t grad_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
        return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
            this, grad_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& segment_grads = Input(SEGMENT_GRADS);
        auto& segment_ids = Input(SEGMENT_IDS);

        CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
        int64_t N = segment_ids.size(0);

        typename ReducerGradient::Meta ctx(segment_grads, 1);
        for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
          auto& aux_in = Input(i);
          CAFFE_ENFORCE_EQ(
              N,
              aux_in.size(0),
              "Input ",
              i,
              " must have the same first dim as SEGMENT_IDS");
          ctx.observeOriginalInput(
              ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
        }

        const SIndex* s_ids = segment_ids.template data<SIndex>();
        const T* s_grads = segment_grads.template data<T>();

        vector<int64_t> shape;
        shape.push_back(N);
        ctx.appendGradShape(&shape);
        auto* data_grads = Output(0, shape, at::dtype<T>());

        int64_t d_block_size = data_grads->size_from_dim(1);
        const SIndex K = segment_grads.size(0);
        int64_t s_block_size = segment_grads.size_from_dim(1);
        T* out = data_grads->template mutable_data<T>();

        if (N == 0) {
          return true;
        }

        // Assume the segments are sorted and there are no gaps
        CAFFE_ENFORCE_EQ(0, s_ids[0], "Indices must be sorted and not have gaps");
        // repeat the check from forward op
        CAFFE_ENFORCE_EQ(
            K - 1, s_ids[N - 1], "Indices must be sorted and not have gaps");
        for (int64_t i = 0; i < N;) {
          int64_t start = i;
          int64_t end = start;

          if (ReducerGradient::computeLength()) {
            for (; end < N && s_ids[start] == s_ids[end]; ++end) {
            }
          }

          ReducerGradient r(ctx, s_grads + s_block_size * s_ids[start], &context_);
          for (; i < N && s_ids[start] == s_ids[i]; ++i) {
            r.template fillGrad<FixedSize>(
                ctx, out + d_block_size * i, i, &context_, end - start);
          }

          // check correctness of the next segment
          if (i < N) {
            CAFFE_ENFORCE_EQ(
                s_ids[start] + 1,
                s_ids[i],
                "Indices must be sorted and not have gaps");
          }
        }
        return true;
        */
    }
}

/**
  | base implementation of sorted/unsorted
  | sparse/non-sparse gradient computation
  |
  */
pub struct SegmentOpGetGradient<ForwardOp,ReducerDef,ReducerGradient,const Sorted: bool,const SparseFused: bool> {
    phantomForwardOp:       PhantomData<ForwardOp>,
    phantomReducerDef:      PhantomData<ReducerDef>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

impl<
    ForwardOp,
    ReducerDef,
    ReducerGradient,
    const Sorted: bool,
    const SparseFused: bool> 
GetGradientDefs for 
    SegmentOpGetGradient<
        ForwardOp,
        ReducerDef,
        ReducerGradient,
        Sorted,
        SparseFused> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            !ReducerGradient::requiresDataInput(Def()),
            "grads on aux inputs are not yet implemented for Segment operators.");
        vector<string> grad_ins;
        for (const int i : ReducerGradient::originalInputs()) {
          grad_ins.push_back(I(i));
        }
        grad_ins.push_back(GO(0));
        grad_ins.push_back(I(ForwardOp::SEGMENT_IDS));
        vector<OperatorDef> r{CreateOperatorDef(
            string(Sorted ? "SortedSegment" : "UnsortedSegment") +
                ReducerDef::name + "Gradient",
            "",
            grad_ins,
            // no gradient on segment_ids or auxiliary inputs for now
            vector<string>{SparseFused ? GI_V(0) : GI(0)})};
        if (SparseFused) {
          SetSparse(0, I(ForwardOp::INDICES), GI_V(0));
        }
        return r;
        */
    }
}

/**
  | Applies '{op}' to each segment of input
  | tensor. Segments need to be sorted and
  | contiguous. See also UnsortedSegment{op}
  | that doesn't have this requirement.
  | 
  | SEGMENT_IDS is a vector that maps each
  | of the first dimension slices of the
  | 
  | DATA to a particular group (segment).
  | Values belonging to the same segment
  | are aggregated together.
  | 
  | The first dimension of the output is
  | equal to the number of input segments,
  | i.e. `SEGMENT_IDS[-1]+1`. Other dimensions
  | are inherited from the input tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractSortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "SortedSegment";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient =
      |     typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractSortedSegmentOp<T, SIndex, Context, Reducer, false>;
      |
      | using BackwardOp =
      |     AbstractSortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
      |
      | using GetGradient = SegmentOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     true /*Sorted*/,
      |     false /*SparseFused*/>;
    */
        phantom:           PhantomData<Context>,
        phantomT:          PhantomData<T>,
        phantomSIndex:     PhantomData<SIndex>,
        phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSortedSegmentDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "SEGMENT_IDS",
                "Vector with the same length as the first dimension of DATA "
                "and values in the range 0..K-1 and in increasing order that "
                "maps each slice of DATA to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of K "
                "(the number of segments).");
            ReducerDef::PopulateSchema(schema);
        */
    }
}

/**
  | Pulls in slices of the input tensor,
  | groups them into segments and applies
  | '{op}' to each segment. Segments need
  | to be sorted and contiguous. See also
  | 
  | SparseUnsortedSegment{op} that doesn't
  | have this requirement.
  | 
  | This op is basically Gather and SortedSegment{op}
  | fused together.
  | 
  | INDICES should contain integers in
  | range 0..N-1 where N is the first dimension
  | of DATA. INDICES represent which slices
  | of DATA need to be pulled in.
  | 
  | SEGMENT_IDS is a vector that maps each
  | referenced slice of the DATA to a particular
  | group (segment). Values belonging
  | to the same segment are aggregated together.
  | SEGMENT_IDS should have the same dimension
  | as INDICES.
  | 
  | The first dimension of the output is
  | equal to the number of input segments,
  | i.e. `SEGMENT_IDS[-1]+1`. Other dimensions
  | are inherited from the input tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractSparseSortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "SparseSortedSegment";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient =
      |     typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractSortedSegmentOp<T, SIndex, Context, Reducer>;
      |
      | // TODO(dzhulgakov): we're registering the same class twice here,
      | // consider avoiding op duplication here
      | using BackwardOp = AbstractSortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
      |
      | using GetGradient = SegmentOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     true /*Sorted*/,
      |     true /*SparseFused*/>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSparseSortedSegmentDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "INDICES",
                "Integer vector containing indices of the first dimension of DATA for "
                "the slices that are being aggregated");
            schema.Input(
                <R as Reducer>::InputCount + 1,
                "SEGMENT_IDS",
                "Vector with the same length as INDICES and values in the range "
                "0..K-1 and in increasing order that maps each slice of DATA referenced"
                " by INDICES to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of K "
                "(the number of segments).");
            ReducerDef::PopulateSchema(schema);
        */
    }
}

/**
 | @brief Unsorted segment reduction op with
 | optional fused embedding lookup
 |
 | Base implementation for UnsortedSegmentXXX and
 | UnsparseSortedSegmentXXX depending on
 |  SparseFused static argument.
 |
 | Unlike the sorted version it allows to have
 | "gaps" in segment ids.
 |
 | Inputs:
 |   0: DATA - input embedding to do lookups in
 |   1..P: AUX_ARG_<I> - optional additional arguments to be passed to the
 |                       reducer, should have the same first dimension as
 |                       SEGMENT_IDS (e.g. scalars in WeightedSum)
 |   # if SparseFused == true:
 |   P+1: INDICES - 1-D vector with indices to look up in DATA. Should have the
 |                  same dimension as SEGMENT_IDS
 |   # P+1 if SparseFused == false:
 |   P+1 or P+2: SEGMENT_IDS - unsorted segment ids 1-D vector
 |
 | Args:
 |   num_segments - allows to override the
 |   dimension of the output. If not set it would
 |   be inferred from segment_ids tensor.
 |
 |
 | Output:
 |   Tensor with first dimension of K, where K is
 |   the max segment id + 1. Rest of dimensions are
 |   decided by reducer but usually are the same
 |   size as extra dimensions of DATA
 |
 |  bool SparseFused = true,
 |  class InputAccessor = BaseInputAccessor<T>>
 */
pub struct AbstractUnsortedSegmentOp<T,SIndex,Context,Reducer,const SparseFused: bool,InputAccessor> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,

    /// member field to reuse memory
    reducers:        Vec<Reducer>,

    input_accessor:  InputAccessor,
    num_segments:    i64,

    phantom: PhantomData<T>,
    phantomSIndex: PhantomData<SIndex>,
}

/**
  | TODO: what do the below two comments
  | *break*, if anything?
  |
  */
pub enum AbstractUnsortedSegmentOpInputTags {

    Indices,   // = <R as Reducer>::InputCount,
    SegmentIds,// = <R as Reducer>::InputCount + ternary![SparseFused,1,0]
}

impl<T, SIndex, Context, R: Reducer, const SparseFused: bool, InputAccessor> 
AbstractUnsortedSegmentOp<T,SIndex,Context,R,SparseFused,InputAccessor> {

    const kSelfInputs: isize = ternary![SparseFused,2,1];
    const kNumInputs:  isize = <R as Reducer>::InputCount + Self::kSelfInputs;

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "num_segments", num_segments_, -1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (SparseFused) {
          return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
              this, Input(INDICES));
        } else {
          // type doesn't matter
          return DoRunWithType<int64_t>();
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self) -> bool {
    
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t in_block_size = Input(0).size_from_dim(1);
        return DispatchHelper<typename Reducer::FixedDispatch, IndexType>::call(
            this, in_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<IndexType, const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& data = Input(0);
        auto& segment_ids = Input(SEGMENT_IDS);

        CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
        int64_t N = segment_ids.size(0);
        const int64_t M = data.size(0);

        const IndexType* idxs;
        if (SparseFused) { // static if
          auto& indices = Input(INDICES);
          CAFFE_ENFORCE_EQ(1, indices.dim(), "INDICES must be a vector");
          CAFFE_ENFORCE_EQ(
              N,
              indices.size(0),
              "SEGMENT_IDS must have the same length as INDICES");
          idxs = indices.template data<IndexType>();
        } else {
          CAFFE_ENFORCE_EQ(
              N, M, "DATA must have the same first dimension as SEGMENT_IDS");
        }

        // It would probably look nicer with varargs templates but it's too much
        // metaprogramming
        typename Reducer::Meta ctx;
        ctx.observeInput(0, data, 1);
        for (int i = 1; i < <R as Reducer>::InputCount; ++i) {
          auto& aux_in = Input(i);
          CAFFE_ENFORCE_EQ(
              N,
              aux_in.size(0),
              "Input ",
              i,
              " must have the same first dim as SEGMENT_IDS");
          ctx.observeInput(i, aux_in, 1);
        }

        const SIndex* s_ids = segment_ids.template data<SIndex>();
        OPERATOR_NEEDS_FEATURE(
            inputAccessor_.observeInput(data),
            "Unsupported input type: ",
            data.dtype().name(),
            ".");

        // determine the number of segments
        SIndex K;
        if (num_segments_ != -1) {
          K = num_segments_;
        } else {
          K = 0;
          for (int64_t i = 0; i < N; ++i) {
            K = std::max(K, s_ids[i] + 1);
          }
        }

        vector<int64_t> shape;
        shape.push_back(K);
        ctx.appendOutputShape(&shape);
        auto* output = Output(0, shape, at::dtype<T>());

        int64_t in_block_size = data.size_from_dim(1);
        int64_t out_block_size = output->size_from_dim(1);
        T* out = output->template mutable_data<T>();

        reducers_.clear();
        reducers_.reserve(K);
        for (int64_t i = 0; i < K; ++i) {
          reducers_.emplace_back(ctx, out + out_block_size * i, &context_);
        }

        for (int64_t i = 0; i < N; ++i) {
          auto s_id = s_ids[i];
          CAFFE_ENFORCE(
              0 <= s_id && s_id < K,
              "Segment id out of range: ",
              s_id,
              ", range 0 to ",
              K);
          IndexType idx;
          if (SparseFused) { // static if
            CAFFE_ENFORCE(
                0 <= idxs[i] && idxs[i] < M,
                "Index out of bounds: ",
                idxs[i],
                ", range 0 to ",
                M);
            idx = idxs[i];
          } else {
            idx = i;
          }
          reducers_[s_id].template process<FixedSize>(
              ctx, inputAccessor_.getBlockPtr(in_block_size, idx), i, &context_);
        }

        for (int64_t i = 0; i < K; ++i) {
          reducers_[i].template finish<FixedSize>(ctx, &context_);
        }
        // call reducers destructors (if there is any)
        reducers_.clear();
        return true;
        */
    }
}

/**
  | Gradient actually doesn't depend on
  | whether sparse lookup is fused or not
  |
  */
pub struct AbstractUnsortedSegmentGradientOp<T,SIndex,Context,ReducerGradient> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:        OperatorStorage,
    context:        Context,

    /// member field to reuse memory
    reducers:       Vec<ReducerGradient>,
    segment_length: Vec<i32>,
    phantom:        PhantomData<T>,
    phantomSIndex:  PhantomData<SIndex>,
}

/**
| Input layout:
|
|   orig_arg1, orig_arg2, ..., orig_argN,
|   SEGMENT_GRADS, SEGMENT_IDS
|
| orig_argXs represent original op's inputs and will
| be passed to the reducer directly
|
| TODO: does the below comment break something?
*/
pub enum AbstractUnsortedSegmentGradientOpInputTags {
    SEGMENT_GRADS,// = <ReducerGradient as HasOriginalInputs>::original_inputs_size(),
    SEGMENT_IDS
}

impl<T,SIndex,Context, ReducerGradient: HasOriginalInputs> AbstractUnsortedSegmentGradientOp<T,SIndex,Context,ReducerGradient> {

    const kNumInputs: isize = todo!(); // <ReducerGradient as HasOriginalInputs>::original_inputs_size() + 2;
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t grad_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
        return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
            this, grad_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& segment_grads = Input(SEGMENT_GRADS);
        auto& segment_ids = Input(SEGMENT_IDS);

        CAFFE_ENFORCE_EQ(1, segment_ids.dim(), "SEGMENT_IDS must be a vector");
        int64_t N = segment_ids.size(0);

        typename ReducerGradient::Meta ctx(segment_grads, 1);
        for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
          auto& aux_in = Input(i);
          CAFFE_ENFORCE_EQ(
              N,
              aux_in.size(0),
              "Input ",
              i,
              " must have the same first dim as SEGMENT_IDS");
          ctx.observeOriginalInput(
              ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
        }

        const SIndex* s_ids = segment_ids.template data<SIndex>();
        const T* s_grads = segment_grads.template data<T>();

        vector<int64_t> shape;
        shape.push_back(N);
        ctx.appendGradShape(&shape);
        auto* data_grads = Output(0, shape, at::dtype<T>());

        int64_t d_block_size = data_grads->size_from_dim(1);
        const SIndex K = segment_grads.size(0);
        int64_t s_block_size = segment_grads.size_from_dim(1);
        T* out = data_grads->template mutable_data<T>();

        if (ReducerGradient::computeLength()) {
          segment_length_.resize(K, 0);
          for (int i = 0; i < N; ++i) {
            auto s_id = s_ids[i];
            CAFFE_ENFORCE(
                0 <= s_id && s_id < K,
                "Segment id out of range: ",
                s_id,
                ", range 0 to ",
                K);
            segment_length_[s_ids[i]]++;
          }
        }

        reducers_.clear();
        reducers_.reserve(K);
        for (SIndex i = 0; i < K; ++i) {
          reducers_.emplace_back(ctx, s_grads + s_block_size * i, &context_);
        }

        for (int64_t i = 0; i < N; ++i) {
          auto s_id = s_ids[i];
          if (ReducerGradient::computeLength()) {
            reducers_[s_id].template fillGrad<FixedSize>(
                ctx, out + d_block_size * i, i, &context_, segment_length_[s_id]);
          } else {
            reducers_[s_id].template fillGrad<FixedSize>(
                ctx, out + d_block_size * i, i, &context_, 0);
          }
        }
        // call reducers destructors (if there is any)
        reducers_.clear();
        return true;
        */
    }
}

/**
  | Applies '{op}' to each segment of input
  | tensor. Segments ids can appear in arbitrary
  | order (unlike in SortedSegment{op}).
  | 
  | SEGMENT_IDS is a vector that maps each
  | of the first dimension slices of the
  | 
  | DATA to a particular group (segment).
  | Values belonging to the same segment
  | are aggregated together.
  | 
  | If `num_segments` argument is passed
  | it would be used as a first dimension
  | for the output. Otherwise, it'd be dynamically
  | calculated from as the max value of
  | 
  | SEGMENT_IDS plus one. Other output
  | dimensions are inherited from the input
  | tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "UnsortedSegment";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractUnsortedSegmentOp<
      |     T,
      |     SIndex,
      |     Context,
      |     typename ReducerDef::template Reducer<T, Context>,
      |     false>;
      |
      | using BackwardOp =
      |     AbstractUnsortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
      |
      | using GetGradient = SegmentOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     false /*Sorted*/,
      |     false /*SparseFused*/>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Arg(
                "num_segments",
                "Optional int argument specifying the number of output segments and "
                "thus the first dimension of the output");
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "SEGMENT_IDS",
                "Integer vector with the same length as the first dimension of DATA "
                "that maps each slice of DATA to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of equal to the "
                "number of segments.");
            ReducerDef::PopulateSchema(schema);
        */
    }

}

/**
  | Pulls in slices of the input tensor,
  | groups them into segments and applies
  | '{op}' to each segment. Segments ids
  | can appear in arbitrary order (unlike
  | in
  | 
  | SparseSortedSegment{op}).
  | 
  | This op is basically Gather and UnsortedSegment{op}
  | fused together.
  | 
  | INDICES should contain integers in
  | range 0..N-1 where N is the first dimension
  | of DATA. INDICES represent which slices
  | of DATA need to be pulled in.
  | 
  | SEGMENT_IDS is a vector that maps each
  | referenced slice of the DATA to a particular
  | group (segment). Values belonging
  | to the same segment are aggregated together.
  | SEGMENT_IDS should have the same dimension
  | as INDICES.
  | 
  | If `num_segments` argument is passed
  | it would be used as a first dimension
  | for the output. Otherwise, it'd be dynamically
  | calculated from as the max value of
  | 
  | SEGMENT_IDS plus one. Other output
  | dimensions are inherited from the input
  | tensor.
  | 
  | {op_doc}
  |
  */
pub struct AbstractSparseUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {
    
    /*
    | using OpDef = ReducerDef;
    |
    | static constexpr const char* basename = "SparseUnsortedSegment";
    |
    | using Reducer = typename ReducerDef::template Reducer<T, Context>;
    |
    | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
    |
    | using ForwardOp = AbstractUnsortedSegmentOp<T, SIndex, Context, Reducer>;
    | // TODO(dzhulgakov): we're registering the same class twice here,
    | // consider avoiding op duplication here
    | using BackwardOp = AbstractUnsortedSegmentGradientOp<T, SIndex, Context, ReducerGradient>;
    |
    | using GetGradient = SegmentOpGetGradient<
    |     ForwardOp,
    |     ReducerDef,
    |     ReducerGradient,
    |     false /*Sorted*/,
    |     true /*SparseFused*/>;
        */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef> AbstractSparseUnsortedSegmentDef<T,SIndex,Context,ReducerDef> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "INDICES",
                "Integer vector containing indices of the first dimension of DATA for "
                "the slices that are being aggregated");
            schema.Input(
                <R as Reducer>::InputCount + 1,
                "SEGMENT_IDS",
                "Integer vector with the same length as INDICES that maps each slice "
                "of DATA referenced by INDICES to one of the segments");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of equal to the "
                "number of segments.");
            ReducerDef::PopulateSchema(schema);
        */
    }
}

/**
 | @brief Segment reduction op with optional fused
 | embedding lookup
 |
 | Base implementation for LengthsXXX and
 | SparseLengthsXXX depending on SparseFused static
 |  argument.
 |
 | Inputs:
 |   0: DATA - input embedding to do lookups in
 |   1..P: AUX_ARG_<I> - optional additional arguments to be passed to the
 |                       reducer, should have the same first dimension as
 |                       LENGTHS (e.g. scalars in WeightedSum)
 |   # if SparseFused == true:
 |   P+1: INDICES - 1-D vector with indices to look up in DATA. Should have the
 |                  same dimension as LENGTHS
 |   # P+1 if SparseFused == false:
 |   P+1 or P+2: LENGTHS - lengths on indecies vector
 |
 | Output:
 |   Tensor with first dimension of K, where K = len(LENGTHS). Rest
 |   of dimensions are decided by reducer but usually are the same size as extra
 |   dimensions of DATA
 |
 |    bool SparseFused = true,
 |   class InputAccessor = BaseInputAccessor<TData>>
 |
 | TODO(dzhulgakov): for now it's implemented with
 | incremental reducers because of fused sparse
 | support. But using "lengths" representation
 | actually implies continuous segments and thus
 | range reducers can be used for non-sparse
 | version.
 */

pub struct AbstractLengthsOp<TData,TLengths,Context,R: Reducer,const SparseFused: bool,InputAccessor> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage:         OperatorStorage,
    context:         Context,
    input_accessor:  InputAccessor,
    phantom:         PhantomData<TData>,
    phantomTLengths: PhantomData<TLengths>,
    phantomR:        PhantomData<R>,
}

/**
  | figure out what the two comments below
  | *break*, if anything
  |
  */
pub enum AbstractLengthsOpInputTags {
    INDICES,// = <R as Reducer>::InputCount,
    LENGTHS,// = <R as Reducer>::InputCount + ternary![SparseFused,1,0],
}

impl<TData,TLengths,Context,R: Reducer,const SparseFused: bool,InputAccessor> 
AbstractLengthsOp<TData,TLengths,Context,R,SparseFused,InputAccessor> {

    const kSelfInputs: isize = ternary![SparseFused, 2, 1];
    const kNumInputs:  isize = <R as Reducer>::InputCount + Self::kSelfInputs;
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (SparseFused) {
          return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
              this, Input(INDICES));
        } else {
          // type doesn't matter
          return DoRunWithType<int64_t>();
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self) -> bool {
    
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t in_block_size = Input(0).size_from_dim(1);
        return DispatchHelper<typename Reducer::FixedDispatch, IndexType>::call(
            this, in_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<IndexType, const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& dataInput = Input(0);
        auto& lengthsInput = Input(LENGTHS);

        CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
        const int64_t dataSize = dataInput.size(0);
        // Either first dim the data or how much we pull in indexies from it
        int64_t dataToReduceSize;
        const int64_t outputSize = lengthsInput.size(0);

        const IndexType* indices;
        if (SparseFused) { // static if
          auto& indicesInput = Input(INDICES);
          CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
          indices = indicesInput.template data<IndexType>();
          dataToReduceSize = indicesInput.size(0);
        } else {
          dataToReduceSize = dataSize;
        }

        typename Reducer::Meta ctx;
        ctx.observeInput(0, dataInput, 1);
        for (int i = 1; i < <R as Reducer>::InputCount; ++i) {
          auto& aux_in = Input(i);
          CAFFE_ENFORCE(
              dataToReduceSize == aux_in.size(0),
              "Input ",
              i,
              " must have the same first dim as SEGMENT_IDS");
          ctx.observeInput(i, aux_in, 1);
        }

        const TLengths* lengths = lengthsInput.template data<TLengths>();

        OPERATOR_NEEDS_FEATURE(
            inputAccessor_.observeInput(dataInput),
            "Unsupported input type: ",
            dataInput.dtype().name(),
            ".");

        vector<int64_t> shape{outputSize};
        ctx.appendOutputShape(&shape);
        auto* output = Output(0, shape, at::dtype<TData>());

        int64_t in_block_size = dataInput.size_from_dim(1);
        int64_t out_block_size = output->size_from_dim(1);
        TData* out = output->template mutable_data<TData>();

        int64_t dataIndex = 0;
        for (int64_t rangeIndex = 0; rangeIndex < outputSize; ++rangeIndex) {
          Reducer reducer(ctx, out + out_block_size * rangeIndex, &context_);
          for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            IndexType idx;
            if (SparseFused) { // static if
              idx = indices[dataIndex];
              CAFFE_ENFORCE(
                  0 <= idx && idx < dataSize,
                  "The ",
                  dataIndex,
                  "th index from the input indices is out of bounds: ",
                  idx,
                  " vs. valid range 0 to ",
                  dataSize);
            } else {
              idx = dataIndex;
              CAFFE_ENFORCE(
                  0 <= idx && idx < dataSize,
                  "When calculating the ",
                  rangeIndex,
                  "th output with length=",
                  lengths[rangeIndex],
                  ", the index is out of bounds: ",
                  idx,
                  " vs. valid range 0 to ",
                  dataSize);
            }

            const TData* input = inputAccessor_.getBlockPtr(in_block_size, idx);
            reducer.template process<FixedSize>(ctx, input, dataIndex, &context_);
          }
          reducer.template finish<FixedSize>(ctx, &context_);
        }
        CAFFE_ENFORCE(
            dataIndex == dataToReduceSize, dataIndex, " != ", dataToReduceSize);

        return true;
        */
    }
}

/**
  | Some notice:
  | 
  | 1. Gradient actually doesn't depend
  | on whether sparse lookup is fused or
  | not
  | 
  | 2. INDICES are not used in CPU version,
  | but they are needed in async CUDA version.
  | So we register 3 input version for CPU
  | as gradient op for
  | 
  | GPU/CPU convert. We then register 2
  | input version for CPU for backward compatibility
  | with older nets.
  | 
  | bool GradientNeedIndices = false>
  |
  */
pub struct AbstractLengthsGradientOp<T,TLengths,Context,ReducerGradient,const GradientNeedIndices: bool> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage:                OperatorStorage,
    context:                Context,

    /*
    | // Input layout:
    | //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, LENGTHS, INDICES
    | // orig_argXs represent original op's inputs and will be passed to the reducer
    | // directly
    | static constexpr int kNumInputs = ReducerGradient::originalInputs().size() + 2 + (GradientNeedIndices ? 1 : 0);
    | enum AbstractLengthsGradientOp_InputTags {
    |     SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
    |     LENGTHS,
    |     INDICES
    | }
    */
    phantom:                PhantomData<T>,
    phantomTLengths:        PhantomData<TLengths>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

impl<T,TLengths,Context,ReducerGradient,const GradientNeedIndices: bool> 
AbstractLengthsGradientOp<T,TLengths,Context,ReducerGradient,GradientNeedIndices> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t gradBlockSize = Input(SEGMENT_GRADS).size_from_dim(1);
        return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
            this, gradBlockSize);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& segmentGradsInput = Input(SEGMENT_GRADS);
        auto& lengthsInput = Input(LENGTHS);

        CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
        int64_t reducedDataSize = 0;
        int64_t numSegments = lengthsInput.size(0);
        CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
        CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
        const TLengths* lengths = lengthsInput.template data<TLengths>();
        for (int64_t i = 0; i < numSegments; ++i) {
          reducedDataSize += lengths[i];
        }

        typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
        for (auto i = 0U; i < ReducerGradient::originalInputs().size(); ++i) {
          auto& aux_in = Input(i);
          CAFFE_ENFORCE_EQ(
              reducedDataSize,
              aux_in.size(0),
              "Input ",
              i,
              " must have the same first dim as SEGMENT_IDS");
          ctx.observeOriginalInput(
              ReducerGradient::originalInputs()[i], aux_in, nullptr /*no grad*/, 1);
        }

        const T* segmentGrads = segmentGradsInput.template data<T>();

        vector<int64_t> shape;
        shape.push_back(reducedDataSize);
        ctx.appendGradShape(&shape);
        auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

        int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
        int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
        T* dataGrads = dataGradsOutput->template mutable_data<T>();

        int64_t dataIndex = 0;
        for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          ReducerGradient reducer(
              ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
          for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            reducer.template fillGrad<FixedSize>(
                ctx,
                dataGrads + dataGradsBlockSize * dataIndex,
                dataIndex,
                &context_,
                lengths[rangeIndex]);
          }
        }
        CAFFE_ENFORCE(
            dataIndex == reducedDataSize, dataIndex, " != ", reducedDataSize);
        return true;
        */
    }
}

/**
 | Version of gradient that requires the main input
 | and thus needs to receive length, indices and
 | other stuff
 |
 | bool SparseFused = true,
 | bool GradientNeedIndices = false>
 */
pub struct AbstractLengthsWithMainInputGradientOp<Tembedding,T,TLengths,Context,ReducerGradient,const SparseFused: bool,const GradientNeedIndices: bool> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,

    /*
     | // Input layout:
     | //   orig_arg1, orig_arg2, ..., orig_argN, SEGMENT_GRADS, LENGTHS,
     | //      DATA_INPUT, [INDICES]
     | // orig_argXs represent original op's inputs and will be passed to the reducer
     | // directly
     | static constexpr int kNumInputs = ReducerGradient::originalInputs().size() + 3 + (SparseFused ? 1 : 0) + (GradientNeedIndices ? 1 : 0);
     | enum _InputTags {
     |     SEGMENT_GRADS = ReducerGradient::originalInputs().size(),
     |     LENGTHS,
     |     DATA_INPUT,
     |     INDICES,
     | };
     */
    phantom:                PhantomData<T>,
    phantomTembedding:      PhantomData<Tembedding>,
    phantomTLengths:        PhantomData<TLengths>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

impl<Tembedding,T,TLengths,Context,ReducerGradient,const SparseFused: bool,const GradientNeedIndices: bool> 
AbstractLengthsWithMainInputGradientOp<Tembedding,T,TLengths,Context,ReducerGradient,SparseFused,GradientNeedIndices> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (SparseFused) {
          return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
              this, Input(INDICES));
        } else {
          // type doesn't matter
          return DoRunWithType<int64_t>();
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self) -> bool {
    
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t in_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
        return DispatchHelper<typename ReducerGradient::FixedDispatch, IndexType>::
            call(this, in_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<IndexType, const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& dataInput = Input(DATA_INPUT);
        auto& segmentGradsInput = Input(SEGMENT_GRADS);
        auto& lengthsInput = Input(LENGTHS);

        CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
        int64_t numSegments = lengthsInput.size(0);
        CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
        CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
        const TLengths* lengths = lengthsInput.template data<TLengths>();

        typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
        for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
          int aux_num = ReducerGradient::originalInputs()[i];
          auto& aux_in = Input(i);
          auto* aux_grad = aux_num < OutputSize() ? Output(aux_num) : nullptr;
          ctx.observeOriginalInput(aux_num, aux_in, aux_grad, 1);
        }

        // Either first dim the data or how much we pull in indexies from it
        int64_t dataToReduceSize;
        const IndexType* indices = nullptr;
        if (SparseFused) { // static if
          auto& indicesInput = Input(INDICES);
          indices = indicesInput.template data<IndexType>();
          dataToReduceSize = indicesInput.size(0);
        } else {
          dataToReduceSize = dataInput.size(0);
        }

        const T* segmentGrads = segmentGradsInput.template data<T>();

        vector<int64_t> shape;
        shape.push_back(dataToReduceSize);
        ctx.appendGradShape(&shape);
        auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

        int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
        int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
        T* dataGrads = dataGradsOutput->template mutable_data<T>();

        const Tembedding* data = dataInput.template data<Tembedding>();
        int64_t dataIndex = 0;
        for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          ReducerGradient reducer(
              ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
          for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            IndexType data_pos;
            // No range checking, should've been verified in forward pass
            if (SparseFused) { // static if
              data_pos = indices[dataIndex];
            } else {
              data_pos = dataIndex;
            }
            reducer.template fillGradWithMainInput<FixedSize>(
                ctx,
                data + dataGradsBlockSize * data_pos,
                dataGrads + dataGradsBlockSize * dataIndex,
                dataIndex,
                &context_,
                lengths[rangeIndex]);
          }
        }
        return true;
        */
    }
}

/**
  | Version of gradient that requires the
  | main input as well as the output of the
  | forward op.
  |
  */
pub struct AbstractLengthsWithMainInputAndForwardOutputGradientOp<T,TLengths,Context,ReducerGradient> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS

    storage: OperatorStorage,
    context: Context,

    /*
    | // Input layout:
    | //   orig_arg1, orig_arg2, ..., orig_argN, FORWARD_OUTPUT, SEGMENT_GRADS,
    | //      LENGTHS, DATA_INPUT
    | // orig_argXs represent original op's inputs and will be passed to the reducer
    | // directly
    | static constexpr int kNumInputs =
    |     ReducerGradient::originalInputs().size() + 4;
    | enum _InputTags {
    |   FORWARD_OUTPUT = ReducerGradient::originalInputs().size(),
    |   SEGMENT_GRADS,
    |   LENGTHS,
    |   DATA_INPUT,
    | };
    */
    phantom:                PhantomData<T>,
    phantomTLengths:        PhantomData<TLengths>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

impl<T,TLengths,Context,ReducerGradient> AbstractLengthsWithMainInputAndForwardOutputGradientOp<T,TLengths,Context,ReducerGradient> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class.
        int64_t in_block_size = Input(SEGMENT_GRADS).size_from_dim(1);
        return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
            this, in_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& dataInput = Input(DATA_INPUT);
        auto& segmentGradsInput = Input(SEGMENT_GRADS);
        auto& lengthsInput = Input(LENGTHS);
        auto& forwardOutputInput = Input(FORWARD_OUTPUT);

        CAFFE_ENFORCE(lengthsInput.dim() == 1, "LENGTHS must be a vector");
        int64_t numSegments = lengthsInput.size(0);
        CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
        CAFFE_ENFORCE(numSegments == segmentGradsInput.size(0));
        const TLengths* lengths = lengthsInput.template data<TLengths>();

        typename ReducerGradient::Meta ctx(segmentGradsInput, 1);
        for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
          int aux_num = ReducerGradient::originalInputs()[i];
          auto& aux_in = Input(i);
          auto* aux_grad = aux_num < OutputSize() ? Output(aux_num) : nullptr;
          ctx.observeOriginalInput(aux_num, aux_in, aux_grad, 1);
        }

        CAFFE_ENFORCE(forwardOutputInput.dim() > 0);
        CAFFE_ENFORCE(numSegments == forwardOutputInput.size(0));
        const T* forwardOutput = forwardOutputInput.template data<T>();

        int64_t dataToReduceSize = dataInput.size(0);

        const T* segmentGrads = segmentGradsInput.template data<T>();

        vector<int64_t> shape;
        shape.push_back(dataToReduceSize);
        ctx.appendGradShape(&shape);
        auto* dataGradsOutput = Output(0, shape, at::dtype<T>());

        int64_t dataGradsBlockSize = dataGradsOutput->size_from_dim(1);
        int64_t segmentBlockSize = segmentGradsInput.size_from_dim(1);
        T* dataGrads = dataGradsOutput->template mutable_data<T>();

        const T* data = dataInput.template data<T>();

        int64_t dataIndex = 0;
        for (int64_t rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          ReducerGradient reducer(
              ctx, segmentGrads + segmentBlockSize * rangeIndex, &context_);
          for (int64_t start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            // No range checking, should've been verified in forward pass
            reducer.template fillGradWithMainInputAndForwardOutput<FixedSize>(
                ctx,
                data + dataGradsBlockSize * dataIndex,
                dataGrads + dataGradsBlockSize * dataIndex,
                forwardOutput + segmentBlockSize * rangeIndex,
                dataIndex,
                &context_,
                lengths[rangeIndex]);
          }
        }
        return true;
        */
    }
}

/**
  | base implementation of sparse/non-sparse
  | gradient computation
  |
  | bool GradientNeedIndices = false>
  |
  */
pub struct LengthsOpGetGradient<ForwardOp,ReducerDef,ReducerGradient,const SparseFused: bool,const GradientNeedIndices: bool> {
    phantomForwardOp:       PhantomData<ForwardOp>,
    phantomReducerDef:      PhantomData<ReducerDef>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

impl<
    ForwardOp,
    ReducerDef,
    ReducerGradient,
    const SparseFused: bool,
    const GradientNeedIndices: bool> 
GetGradientDefs 
    for LengthsOpGetGradient<
        ForwardOp,
        ReducerDef,
        ReducerGradient,
        SparseFused,
        GradientNeedIndices> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<string> grad_ins;
        string suffix = "Gradient";
        for (const int i : ReducerGradient::originalInputs()) {
          grad_ins.push_back(I(i));
        }
        if (ReducerGradient::requiresForwardOutput()) {
          grad_ins.push_back(O(0));
          CAFFE_ENFORCE(
              !SparseFused,
              "Forward pass output not yet supported as input for backward pass "
              "for SparseLengthsXXX operators");
          suffix = "AndForwardOutput" + suffix;
        }
        grad_ins.push_back(GO(0));
        grad_ins.push_back(I(ForwardOp::LENGTHS));
        bool indices_pushed = false;
        if (ReducerGradient::requiresDataInput(Def())) {
          grad_ins.push_back(I(0));
          if (SparseFused) {
            grad_ins.push_back(I(ForwardOp::INDICES));
            indices_pushed = true;
          }
          suffix = "WithMainInput" + suffix;
        }
        if (GradientNeedIndices && !indices_pushed) {
          if (SparseFused) {
            grad_ins.push_back(I(ForwardOp::INDICES));
          } else {
            // Hacky: using Input as Indices, remove this after we have specialized
            // cuda LengthsIndicesInGradientSumGradient
            grad_ins.push_back(I(0));
          }
        }
        vector<string> grad_outs;
        grad_outs.push_back({SparseFused ? GI_V(0) : GI(0)});
        int aux_grads = ReducerGradient::numAuxInputsWithGrads(Def());
        for (int i = 1; i <= aux_grads; ++i) {
          grad_outs.push_back(GI(i));
        }
        vector<OperatorDef> r{CreateOperatorDef(
            string(SparseFused ? "SparseLengths" : "Lengths") +
                string(GradientNeedIndices ? "IndicesInGradient" : "") +
                ReducerDef::name + suffix,
            "",
            grad_ins,
            grad_outs)};
        if (SparseFused) {
          SetSparse(0, I(ForwardOp::INDICES), GI_V(0));
        }
        return r;
        */
    }
}

/**
  | Applies '{op}' to each segment of the
  | input tensor. Segments are defined
  | by their *LENGTHS*. *LENGTHS* is a vector
  | that maps each of the slices of
  | 
  | DATA* to a particular segment. Values
  | belonging to the same segment are aggregated
  | together and considered for the '{op}'
  | operation.
  | 
  | For example *LENGTHS = [2, 1]* stands
  | for segments *DATA[0..1]* and *DATA[2]*
  | 
  | The sum of elements in *LENGTHS* must
  | equal the number of elements in the first
  | dimension of *DATA*. The length of *OUTPUT*
  | is equal to the number of input segments,
  | i.e. len(*LENGTHS*).
  | 
  | {op_doc}
  | 
  | {extra}
  | 
  | bool GradientNeedIndices = false>
  |
  */
pub struct AbstractLengthsDef<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> {
    
    /*
      using OpDef = ReducerDef;

      static constexpr const char* basename = "Lengths";

      using Reducer = typename ReducerDef::template Reducer<T, Context>;

      using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;

      using ForwardOp = AbstractLengthsOp<T, SIndex, Context, Reducer, false>;

      using BackwardOp = AbstractLengthsGradientOp<T, SIndex, Context, ReducerGradient>;

      using WithMainInputBackwardOp = AbstractLengthsWithMainInputGradientOp<
          T,
          T,
          SIndex,
          Context,
          ReducerGradient,
          false>;

      using WithMainInputAndForwardOutputBackwardOp =
          AbstractLengthsWithMainInputAndForwardOutputGradientOp<
              T,
              SIndex,
              Context,
              ReducerGradient>;

      using GetGradient = LengthsOpGetGradient<
          ForwardOp,
          ReducerDef,
          ReducerGradient,
          false /*SparseFused*/,
          GradientNeedIndices>;
    */
        phantom: PhantomData<Context>,
        phantomT: PhantomData<T>,
        phantomSIndex: PhantomData<SIndex>,
        phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> 
AbstractLengthsDef<T,SIndex,Context,ReducerDef,GradientNeedIndices> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "LENGTHS",
                "Vector with the same sum of elements as the first dimension of DATA");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of len(LENGTHS) ");
            schema.TensorInferenceFunction(
                [](const OperatorDef& def, const vector<TensorShape>& in) {
                  vector<TensorShape> out(0);
                  TensorShape output;
                  for (int d : in[<R as Reducer>::InputCount].dims()) {
                    output.add_dims(d);
                  }
                  for (int j = 1; j < in[0].dims_size(); j++) {
                    output.add_dims(in[0].dims(j));
                  }
                  output.set_data_type(in[0].data_type());
                  out.push_back(output);
                  return out;
                });
            ReducerDef::PopulateSchema(schema);
        */
    }
}

/**
  | Pulls in slices of the input tensor,
  | groups them into segments and applies
  | '{op}' to each segment. Segments are
  | defined by their LENGTHS.
  | 
  | This op is basically Gather and Lengths{op}
  | fused together.
  | 
  | INDICES should contain integers in
  | range 0..N-1 where N is the first dimension
  | of DATA. INDICES represent which slices
  | of DATA need to be pulled in.
  | 
  | LENGTHS is a vector that defines slice
  | sizes by first dimension of DATA. Values
  | belonging to the same segment are aggregated
  | together. sum(LENGTHS) has to match
  | INDICES size.
  | 
  | The first dimension of the output is
  | equal to the number of input segment,
  | i.e. `len(LENGTHS)`. Other dimensions
  | are inherited from the input tensor.
  | 
  | {op_doc} bool GradientNeedIndices
  | = false>
  |
  */
pub struct AbstractSparseLengthsDef<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> {
    
    /*
      | using OpDef = ReducerDef;
      |
      | static constexpr const char* basename = "SparseLengths";
      |
      | using Reducer = typename ReducerDef::template Reducer<T, Context>;
      |
      | using ReducerGradient = typename ReducerDef::template ReducerGradient<T, Context>;
      |
      | using ForwardOp = AbstractLengthsOp<T, SIndex, Context, Reducer>;
      |
      | // TODO(dzhulgakov): we're registering the same class twice here,
      | // consider avoiding op duplication here
      | // Note: registering 2 input version for now because of naming in the macro,
      | // will register 3 input version alone
      | /* INDICES are not used in CPU version, but they are needed in async CUDA
      |  *    version. So we register 3 input version for CPU as gradient op for
      |  *    GPU/CPU convert. We then register 2 input version for CPU for backward
      |  *    compatibility with older nets.
      |  */
      | using BackwardOp = AbstractLengthsGradientOp<
      |     T,
      |     SIndex,
      |     Context,
      |     ReducerGradient,
      |     false /*GradientNeedIndices*/>;
      |
      | using WithMainInputBackwardOp = AbstractLengthsWithMainInputGradientOp<
      |     T,
      |     T,
      |     SIndex,
      |     Context,
      |     ReducerGradient>;
      |
      | // Will return 3 input version. This is aligning new CPU/GPU nets.
      | using GetGradient = LengthsOpGetGradient<
      |     ForwardOp,
      |     ReducerDef,
      |     ReducerGradient,
      |     true /*SparseFused*/,
      |     GradientNeedIndices>;
    */
    phantomA:          PhantomData<T>,
    phantomB:          PhantomData<Context>,
    phantomSIndex:     PhantomData<SIndex>,
    phantomReducerDef: PhantomData<ReducerDef>,
}

impl<T,SIndex,Context,ReducerDef,const GradientNeedIndices: bool> 
AbstractSparseLengthsDef<T,SIndex,Context,ReducerDef,GradientNeedIndices> {

    #[inline] pub fn populate_schema(schema: &mut OpSchema)  {
        
        todo!();
        /*
            schema.Input(0, "DATA", "Input tensor, slices of which are aggregated.");
            schema.Input(
                <R as Reducer>::InputCount,
                "INDICES",
                "Integer vector containing indices of the first dimension of DATA for "
                "the slices that are being aggregated");
            schema.Input(
                <R as Reducer>::InputCount + 1,
                "LENGTHS",
                "Non negative vector with sum of elements equal to INDICES length");
            schema.Output(
                0,
                "OUTPUT",
                "Aggregated output tensor. Has the first dimension of K "
                "(the number of segments).");
            schema.TensorInferenceFunction(OpSchema::NeedsAllInputShapes(
                [](const OperatorDef&, const std::vector<TensorShape>& input_types) {
                  std::vector<TensorShape> out(1);
                  out[0] = input_types[0];
                  out[0].set_dims(0, input_types[<R as Reducer>::InputCount + 1].dims(0));
                  return out;
                }));
            ReducerDef::PopulateSchema(schema);

            schema.CostInferenceFunction(
                [](const OperatorDef& def,
                   const vector<TensorShape>& inputs) -> OpSchema::Cost {
                  return CostInferenceForSparseLengths(
                      def, inputs, strcmp(OpDef::name, "WeightedSum") == 0);
                });
        */
    }
}

#[inline] pub fn cost_inference_for_sparse_lengths(
    def:        &OperatorDef,
    inputs:     &Vec<TensorShape>,
    use_weight: bool) -> OpSchemaCost {
    
    todo!();
    /*
        int min_num_of_inputs = 3 + use_weight;
      CAFFE_ENFORCE_GE(
          inputs.size(),
          min_num_of_inputs,
          def.type() + " requires at least " + c10::to_string(min_num_of_inputs));

      const TensorShape data = inputs[0];
      const TensorShape indices = inputs[1 + use_weight];
      const TensorShape lengths = inputs[2 + use_weight];

      OpSchema::Cost c;
      CAFFE_ENFORCE_GT(data.dims_size(), 0, "data requires at least 1 dimension");
      uint64_t N = data.dims(0);
      if (N == 0) {
        return c;
      }
      uint64_t D = nElemFromDim(data, 1);
      CAFFE_ENFORCE_GT(
          lengths.dims_size(), 0, "lengths requires at least 1 dimension");
      uint64_t M = lengths.dims(0);
      uint64_t indices_size = nElemFromDim(indices);

      c.flops = indices_size * D;
      c.bytes_read = indices_size *
              (D * sizeof(data.data_type()) + sizeof(indices.data_type())) +
          M * sizeof(lengths.data_type());
      c.params_bytes = N * D * sizeof(data.data_type());
      if (use_weight) {
        const TensorShape weights = inputs[1];
        c.flops += indices_size * D;
        c.bytes_read += indices_size * sizeof(weights.data_type());
      }

      return c;
    */
}

/**
  | registering 5 input gradient with main
  | output gradient of SparseLengthsWeightedSum
  |
  */
num_inputs!{SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient, 5}

num_outputs!{SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient, 2}

register_cpu_operator!{
    SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient,
    AbstractLengthsWithMainInputGradientOp::<
        f32,
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef::ReducerGradient::<f32, CPUContext>,
        SparseFused,
        GradientNeedIndices>
}

/**
  | registering 4 input version
  |
  */
num_inputs!{SparseLengthsIndicesInGradientWeightedSumGradient, 4}

num_outputs!{SparseLengthsIndicesInGradientWeightedSumGradient, 1}

register_cpu_operator!{
    SparseLengthsIndicesInGradientWeightedSumGradient,
    AbstractLengthsGradientOp<
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef::ReducerGradient<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | registering 3 input version gradient
  | of SparseLengthsSum
  |
  */
num_inputs!{SparseLengthsIndicesInGradientSumGradient, 3}

num_outputs!{SparseLengthsIndicesInGradientSumGradient, 1}

register_cpu_operator!{
    SparseLengthsIndicesInGradientSumGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        SumReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | gradient of LengthsSum
  |
  */
num_inputs!{LengthsIndicesInGradientSumGradient, 3}

num_outputs!{LengthsIndicesInGradientSumGradient, 1}

register_cpu_operator!{
    LengthsIndicesInGradientSumGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        SumReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | registering 3 input version gradient
  | of SparseLengthsMean
  |
  */
num_inputs!{SparseLengthsIndicesInGradientMeanGradient, 3}

num_outputs!{SparseLengthsIndicesInGradientMeanGradient, 1}

register_cpu_operator!{
    SparseLengthsIndicesInGradientMeanGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        MeanReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | gradient of LengthsMean
  |
  */
num_inputs!{LengthsIndicesInGradientMeanGradient, 3}

num_outputs!{LengthsIndicesInGradientMeanGradient, 1}

register_cpu_operator!{
    LengthsIndicesInGradientMeanGradient,
    AbstractLengthsGradientOp::<
        f32,
        i32,
        CPUContext,
        MeanReducerDef::ReducerGradient::<f32, CPUContext>,
        GradientNeedIndices>
}

/**
  | The *LengthsMax* op takes two inputs
  | *DATA* and *LENGTHS*, and produces
  | a single output *OUTPUT*.
  | 
  | The op finds the maximum value in each
  | of the segments of *DATA*, where segments
  | are defined by their lengths. For example,
  | if $DATA = [2,4,3,1,2,10]$ and $LENGTHS
  | = [2,3,1]$ then $OUTPUT = [max([2,4]),
  | max([3,1,2]), max([10])] = [4,3,10]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_lengths_ops_main_input_and_forward_output_gradient!{
    LengthsMax,
    LengthsMaxWithMainInputAndForwardOutputGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, MaxReducerDef>
}

pub trait HasForwardOp {
    type ForwardOp;
}

pub type LengthsMaxCPUOp = <AbstractLengthsDef<
    f32,
    i32,
    CPUContext,
    MaxReducerDef,
    true> as HasForwardOp>::ForwardOp;

export_caffe2_op_to_c10_cpu!{
    LengthsMax,
    "_caffe2::LengthsMax(Tensor data, Tensor lengths) -> Tensor",
    LengthsMaxCPUOp
}

#[test] fn lengths_max_extra_op_example() {

    todo!();

    /*

    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsMax",
        ["DATA", "LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [ 4.  3. 10.]

    */
}

/**
  | The *LengthsMean* op takes two inputs
  | *DATA* and *LENGTHS*, and produces
  | a single output *OUTPUT*.
  | 
  | The op finds the mean value in each of
  | the segments of *DATA*, where segments
  | are defined by their lengths. For example,
  | if $DATA = [2,4,3,1,2,10]$ and $LENGTHS
  | = [2,3,1]$ then $OUTPUT = [mean([2,4]),
  | mean([3,1,2]), mean([10])] = [3,2,10]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_segment_def!{
    LengthsMean,
    LengthsMeanGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, MeanReducerDef, true>
}

export_caffe2_op_to_c10_cpu!{
    LengthsMean,
    "_caffe2::LengthsMean(Tensor data, Tensor lengths) -> Tensor",
    LengthsMeanCPUOp}

#[test] fn lengths_mean_extra_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsMean",
        ["DATA", "LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [ 3.  2. 10.]
    */
}

/**
  | The *LengthsSum* op takes two inputs
  | *DATA* and *LENGTHS*, and produces
  | a single output *OUTPUT*.
  | 
  | The op finds the sum in each of the segments
  | of *DATA*, where segments are defined
  | by their lengths. For example, if $DATA
  | = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$
  | then $OUTPUT = [sum([2,4]), sum([3,1,2]),
  | sum([10])] = [6,6,10]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_segment_def!{
    LengthsSum,
    LengthsSumGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, SumReducerDef, true>
}

export_caffe2_op_to_c10_cpu!{
    LengthsSum,
    "_caffe2::LengthsSum(Tensor data, Tensor lengths) -> Tensor",
    LengthsSumCPUOp
}

#[test] fn lengths_sum_extra_op() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsSum",
        ["DATA", "LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [ 6.  6. 10.]
    */
}

/*
| using LengthsSumCPUOp = AbstractLengthsDef<
|     float,
|     int,
|     CPUContext,
|     SumReducerDef,
|     true>::ForwardOp;
|
| using LengthsMeanCPUOp = AbstractLengthsDef<
|     float,
|     int,
|     CPUContext,
|     MeanReducerDef,
|     true>::ForwardOp;
*/

/**
  | The *LengthsWeightedSum* op takes
  | three inputs *DATA*, *LENGTHS*, and
  | *SCALARS*, and produces a single output
  | *OUTPUT*.
  | 
  | The op finds the weighted sum in each
  | of the segments of *DATA*, where segments
  | are defined by their lengths. Before
  | calculating the sums, the input *DATA*
  | is weighted by the contents of *SCALARS*.
  | 
  | For example, if $DATA = [2,4,3,1,2,10]$,
  | $SCALARS = [8, 2, 1, 4, 1, 0.6]$, and $LENGTHS
  | = [2,3,1]$, then $OUTPUT = [sum([8*2,2*4]),
  | sum([1*3,4*1,1*2]), sum([0.6*10])]
  | = [24,9,6]$.
  | 
  | Github Link:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc
  |
  */
register_segment_def!{
    LengthsWeightedSum,
    LengthsWeightedSumGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, WeightedSumReducerDef, false>
}

register_gradient_with_main_input!{
    LengthsWeightedSumWithMainInputGradient,
    AbstractLengthsDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

register_gradient_with_main_input!{
    SparseLengthsWeightedSumWithMainInputGradient,
    AbstractSparseLengthsDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

#[test] fn lengths_weighted_sum_extra_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsWeightedSum",
        ["DATA", "SCALARS","LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("SCALARS", np.array([8, 2, 1, 4, 1, 0.6]).astype(np.float32))
    print("SCALARS:\n", workspace.FetchBlob("SCALARS"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))


    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    SCALARS:
     [8.  2.  1.  4.  1.  0.6]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [24.  9.  6.]
    */
}

///-------------------------------------
#[inline] pub fn format_doc<Def>() -> String {

    todo!();
    /*
        string doc = Def::doc;
      c10::ReplaceAll(doc, "{op}", Def::OpDef::name);
      c10::ReplaceAll(doc, "{op_doc}", Def::OpDef::doc);
      if (strcmp(Def::OpDef::name, "Max") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsMaxExtra);
      } else if (strcmp(Def::OpDef::name, "Mean") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsMeanExtra);
      } else if (strcmp(Def::OpDef::name, "Sum") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsSumExtra);
      } else if (strcmp(Def::OpDef::name, "WeightedSum") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsWeightedSumExtra);
      } else {
        c10::ReplaceAll(doc, "{extra}", " ");
      }
      return doc;
    */
}

/**
  | Helper function to enforce naming conventions
  | at compile time.
  |
  */
#[inline] pub fn equal(
    lhs:  *const u8,
    rhs1: *const u8,
    rhs2: *const u8,
    rhs3: *const u8) -> bool {
    
    todo!();
    /*
        return (*lhs == 0 && *rhs1 == 0 && *rhs2 == 0 && *rhs3 == 0) ||
          (*rhs1 != 0 && *lhs == *rhs1 && equal(lhs + 1, rhs1 + 1, rhs2, rhs3)) ||
          (*rhs1 == 0 && *rhs2 != 0 && *lhs == *rhs2 &&
           equal(lhs + 1, rhs1, rhs2 + 1, rhs3)) ||
          (*rhs1 == 0 && *rhs2 == 0 && *rhs3 != 0 && *lhs == *rhs3 &&
           equal(lhs + 1, rhs1, rhs2, rhs3 + 1));
    */
}

/**
  | Helper macro when the main op is defined
  | elsewhere, and we only need to define the
  | schema, and the gradient op.
  |
  | TODO: enable input fillers
  */
#[macro_export] macro_rules! register_segment_def_schema_gradient_only {
    () => {
        /*
                (                            
            segment_name, gradient_name, ...)                                         
          static_assert(                                                              
              equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name),  
              #segment_name);                                                         
          static_assert(                                                              
              equal(                                                                  
                  #gradient_name,                                                     
                  __VA_ARGS__::basename,                                              
                  __VA_ARGS__::OpDef::name,                                           
                  "Gradient"),                                                        
              #gradient_name);                                                        
          OPERATOR_SCHEMA(segment_name)                                               
              .NumInputs(__VA_ARGS__::ForwardOp::kNumInputs)                          
              .NumOutputs(1)                                                          
              .DisallowInputFillers()                                                 
              .SetDoc(FormatDoc<__VA_ARGS__>())                                       
              .Output(0, "OUTPUT", "Aggregated tensor")                               
              .FillUsing(__VA_ARGS__::PopulateSchema);                                
          REGISTER_CPU_OPERATOR_STR(string(#gradient_name), __VA_ARGS__::BackwardOp); 
          OPERATOR_SCHEMA(gradient_name)                                              
              .NumInputs(__VA_ARGS__::BackwardOp::kNumInputs)                         
              .NumOutputs(1)                                                          
              .DisallowInputFillers();                                                
          REGISTER_GRADIENT_STR(string(#segment_name), __VA_ARGS__::GetGradient)
        */
    }
}

#[macro_export] macro_rules! register_segment_def {
    ($segment_name:expr, $gradient_name:expr, $($arg:expr),*) => {
        /*
        
          static_assert(                                                             
              equal(#segment_name, __VA_ARGS__::basename, __VA_ARGS__::OpDef::name), 
              #segment_name);                                                        
          REGISTER_CPU_OPERATOR_STR(string(#segment_name), __VA_ARGS__::ForwardOp);  
          REGISTER_SEGMENT_DEF_SCHEMA_GRADIENT_ONLY(                                 
              segment_name, gradient_name, __VA_ARGS__)
        */
    }
}

register_segment_def!{
    SortedSegmentRangeSum,
    SortedSegmentRangeSumGradient,
    AbstractSortedSegmentRangeDef::<
        f32, 
        i32, 
        CPUContext, 
        SumRangeReducerDef>
}

register_segment_def!{
    SortedSegmentRangeLogSumExp,
    SortedSegmentRangeLogSumExpGradient,
    AbstractSortedSegmentRangeDef::<
        f32,
        i32,
        CPUContext,
        LogSumExpRangeReducerDef>
}

register_segment_def!{
    SortedSegmentRangeLogMeanExp,
    SortedSegmentRangeLogMeanExpGradient,
    AbstractSortedSegmentRangeDef::<
        f32,
        i32,
        CPUContext,
        LogMeanExpRangeReducerDef>
}

register_segment_def!{
    SortedSegmentRangeMean,
    SortedSegmentRangeMeanGradient,
    AbstractSortedSegmentRangeDef::<f32, i32, CPUContext, MeanRangeReducerDef> 
}

register_segment_def!{
    SortedSegmentRangeMax,
    SortedSegmentRangeMaxGradient,
    AbstractSortedSegmentRangeDef::<f32, i32, CPUContext, MaxRangeReducerDef> 
}

register_segment_def!{
    SortedSegmentSum,
    SortedSegmentSumGradient,
    AbstractSortedSegmentDef::<f32, i32, CPUContext, SumReducerDef> 
}

register_segment_def!{
    SparseSortedSegmentSum,
    SparseSortedSegmentSumGradient,
    AbstractSparseSortedSegmentDef::<f32, i32, CPUContext, SumReducerDef> 
}

register_segment_def!{
    UnsortedSegmentSum,
    UnsortedSegmentSumGradient,
    AbstractUnsortedSegmentDef::<f32, i32, CPUContext, SumReducerDef> 
}

register_segment_def!{
    SparseUnsortedSegmentSum,
    SparseUnsortedSegmentSumGradient,
    AbstractSparseUnsortedSegmentDef::<f32, i32, CPUContext, SumReducerDef>
}

register_segment_def!{
    SortedSegmentMean,
    SortedSegmentMeanGradient,
    AbstractSortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    SparseSortedSegmentMean,
    SparseSortedSegmentMeanGradient,
    AbstractSparseSortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    UnsortedSegmentMean,
    UnsortedSegmentMeanGradient,
    AbstractUnsortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    SparseUnsortedSegmentMean,
    SparseUnsortedSegmentMeanGradient,
    AbstractSparseUnsortedSegmentDef::<f32, i32, CPUContext, MeanReducerDef>
}

register_segment_def!{
    ReduceFrontWeightedSum,
    ReduceFrontWeightedSumGradient,
    AbstractReduceFrontDef::<f32, CPUContext, WeightedSumReducerDef>
}

register_segment_def!{
    SortedSegmentWeightedSum,
    SortedSegmentWeightedSumGradient,
    AbstractSortedSegmentDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

register_segment_def!{
    SparseSortedSegmentWeightedSum,
    SparseSortedSegmentWeightedSumGradient,
    AbstractSparseSortedSegmentDef::<
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef>
}

register_segment_def!{
    UnsortedSegmentWeightedSum,
    UnsortedSegmentWeightedSumGradient,
    AbstractUnsortedSegmentDef::<f32, i32, CPUContext, WeightedSumReducerDef>
}

register_segment_def!{
    SparseUnsortedSegmentWeightedSum,
    SparseUnsortedSegmentWeightedSumGradient,
    AbstractSparseUnsortedSegmentDef::<
        f32,
        i32,
        CPUContext,
        WeightedSumReducerDef>
}

