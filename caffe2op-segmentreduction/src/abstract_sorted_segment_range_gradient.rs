crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_SIMPLE_CTOR_DTOR("AbstractSortedSegmentRangeGradientOp")]
pub struct AbstractSortedSegmentRangeGradientOp<T,SIndex,Context,RangeReducerGradient> {
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
