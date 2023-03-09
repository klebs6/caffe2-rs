crate::ix!();

/**
 | Version of gradient that requires the main input
 | and thus needs to receive length, indices and
 | other stuff
 |
 | bool SparseFused = true,
 | bool GradientNeedIndices = false>
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractLengthsWithMainInputGradientOp<Tembedding,T,TLengths,Context,ReducerGradient,const SparseFused: bool,const GradientNeedIndices: bool> {

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
