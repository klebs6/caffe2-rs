crate::ix!();

/**
  | Version of gradient that requires the
  | main input as well as the output of the
  | forward op.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractLengthsWithMainInputAndForwardOutputGradientOp<T,TLengths,Context,ReducerGradient> {

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


