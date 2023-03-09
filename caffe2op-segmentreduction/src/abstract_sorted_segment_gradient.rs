crate:ix!();

/**
  | Gradient actually doesn't depend on
  | whether sparse lookup is fused or not
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractSortedSegmentGradientOp<T,SIndex,Context,ReducerGradient: HasOriginalInputs> {

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
