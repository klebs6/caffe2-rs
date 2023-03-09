crate::ix!();

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

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractLengthsOp<TData,TLengths,Context,R: Reducer,const SparseFused: bool,InputAccessor> {

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


