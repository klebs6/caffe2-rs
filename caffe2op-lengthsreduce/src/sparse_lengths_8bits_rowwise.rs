crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseLengths8BitsRowwiseOp<Context, const USE_WEIGHTS: bool, const USE_MEAN: bool, OutDataT> {

    storage:         OperatorStorage,
    context:         Context,

    phantomOutDataT: PhantomData<OutDataT>,
}

//OutDataT default type is f32
impl<Context, 
    const USE_WEIGHTS: bool, 
    const USE_MEAN: bool, 
    OutDataT> 
SparseLengths8BitsRowwiseOp<Context, USE_WEIGHTS, USE_MEAN, OutDataT> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self, ) -> bool {
        todo!();
        /*
            auto& dataInput = Input(DATA);
        auto& lengthsInput = Input(LENGTHS);

        auto* scale_bias = Input(SCALE_BIAS).template data<float>();
        CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
        const int64_t outputSize = lengthsInput.size(0);

        auto& indicesInput = Input(INDICES);
        CAFFE_ENFORCE_EQ(2, Input(SCALE_BIAS).dim(), "scale_bias has to be matrix");
        CAFFE_ENFORCE_EQ(
            dataInput.size(0),
            Input(SCALE_BIAS).size(0),
            "scale_bias must have the same first dim as data");
        CAFFE_ENFORCE_EQ(
            2,
            Input(SCALE_BIAS).size(1),
            "the second dim of scale_bias has to be equal to 2");
        CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
        const IndexType* indices = indicesInput.template data<IndexType>();
        int64_t dataToReduceSize = indicesInput.size(0);

        const int* lengths = lengthsInput.template data<int>();
        vector<int64_t> shape = dataInput.sizes().vec();
        shape[0] = outputSize;
        auto* output = Output(0, shape, at::dtype<OutDataT>());
        const float* w = nullptr;
        if (USE_WEIGHTS) {
          w = Input(WEIGHTS).template data<float>();
        }
        int64_t in_block_size = dataInput.size_from_dim(1);
        OutDataT* out = output->template mutable_data<OutDataT>();
        const uint8_t* input_data = dataInput.template data<uint8_t>();

        // delegate work to perfkernel that branches based on architecture
        const int64_t indices_size = indicesInput.numel();
        const int64_t N = dataInput.size(0);
        EmbeddingLookup(
            in_block_size,
            outputSize,
            indices_size,
            N, // embedding table length
            input_data,
            indices,
            lengths,
            w,
            scale_bias,
            USE_MEAN,
            out);

        return true;
        */
    }
}

/**
  |there is weirdness with MaybeWeights it is
  |possible Data is 0, MaybeWeights is 1, Indices
  |is 2, etc or it is possible Data is 0, Indices
  |is 1, etc check the c++ code
  */
pub enum SparseLengths8BitsRowwiseOpIdx {
    Data,
    MaybeWeights,
    Indices,
    Lengths,
    ScaleBias,
}

