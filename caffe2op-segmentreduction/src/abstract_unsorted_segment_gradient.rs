crate::ix!();

/**
  | Gradient actually doesn't depend on
  | whether sparse lookup is fused or not
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractUnsortedSegmentGradientOp<T,SIndex,Context,ReducerGradient> {

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
