crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractSortedSegmentRangeOp<T,SIndex,Context,RangeReducer,InputAccessor> {
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
