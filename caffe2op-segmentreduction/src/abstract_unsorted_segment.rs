crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractUnsortedSegmentOp<T,SIndex,Context,Reducer,const SparseFused: bool,InputAccessor> {

    storage:         OperatorStorage,
    context:         Context,

    /// member field to reuse memory
    reducers:        Vec<Reducer>,

    input_accessor:  InputAccessor,
    num_segments:    i64,

    phantom:         PhantomData<T>,
    phantomSIndex:   PhantomData<SIndex>,
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
