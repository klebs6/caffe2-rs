crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractLengthsGradientOp<T,TLengths,Context,ReducerGradient,const GradientNeedIndices: bool> {

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


