crate::ix!();

/**
  | static_assert( !(with_weights &&
  | is_mean), "Cannot have with_weights
  | and is_mean a the same time");
  |
  */
pub enum SparseLengthsConfig {
    NoWeightsNoMean,
    YesWeightsNoMean,
    YesMeanNoWeights,
}

/**
  |note there is something weird here because we
  |may or may not have a Weights input therefore
  |check how we use Input(N) where N is integer for
  |correct usage (see c++ if this is confusing)
  */
pub enum SparseLengthsFusedNBitRowwiseOpTags {
    Data,
    Weights,
    Indices,
    Lengths,
}

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseLengthsFusedNBitRowwiseOp<const BIT_RATE: i32,Context,const with_weights: bool,const is_mean: bool> {
    storage: OperatorStorage,
    context: Context,

    /*
#ifdef USE_FBGEMM
  std::int64_t last_block_size{-1};
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int32_t>::Type kernel32_;
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int64_t>::Type kernel64_;
#endif
    */
}

impl<const BIT_RATE: i32,Context,const with_weights: bool,const is_mean: bool> 
SparseLengthsFusedNBitRowwiseOp<BIT_RATE,Context,with_weights,is_mean> {
    
    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<IndexType>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& data = Input(DATA);
        const auto& indices = Input(INDICES);
        const auto& lengths = Input(LENGTHS);

        CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES must be a vector");
        CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be a vector");

        const float* weights = nullptr;
        if (with_weights) {
          const auto& weights_input = Input(WEIGHTS);
          CAFFE_ENFORCE_EQ(weights_input.dim(), 1, "WEIGHTS must be a vector");
          CAFFE_ENFORCE_EQ(
              weights_input.numel(),
              indices.numel(),
              "WEIGHTS should have the same length as INDICES.");
          weights = weights_input.template data<float>();
        }

        CAFFE_ENFORCE_GT(
            data.size(1),
            sizeof(at::Half) + sizeof(at::Half),
            "DATA must have more than 4 columns");
        static_assert(8 % BIT_RATE == 0, "BIT_RATE must divide 8");
        constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
        // Subtract 4 from the #columns of data for the 2 bytes for fp16 scale and 2
        // byte for bias that we use in the fused representation (per row).
        const std::vector<int64_t> shape = {
            lengths.size(0),
            static_cast<int64_t>(data.size(1) - 2 * sizeof(at::Half)) *
                NUM_ELEM_PER_BYTE};
        auto* output = Output(0, shape, at::dtype<float>());

        int output_size = output->size(0);
        int block_size = output->size(1);
        CAFFE_ENFORCE_EQ(
            block_size % NUM_ELEM_PER_BYTE,
            0,
            "block size must be divisible by " + std::to_string(NUM_ELEM_PER_BYTE));
        int index_size = indices.numel();
        auto data_size = data.size(0);
        const uint8_t* input_data = data.template data<uint8_t>();
        const IndexType* indices_data = indices.template data<IndexType>();
        const int* lengths_data = lengths.template data<int>();
        float* output_data = output->template mutable_data<float>();

    #ifdef USE_FBGEMM
        // If this is the first call or block size has changed (should never happen
        // actually), generate a kernel.
        if (block_size != last_block_size) {
          last_block_size = block_size;
          if (std::is_same<IndexType, std::int32_t>::value) {
            kernel32_ = fbgemm::GenerateEmbeddingSpMDMNBit<std::int32_t>(
                BIT_RATE,
                block_size,
                weights != nullptr,
                is_mean,
                /*prefetch distance*/ 8,
                /*is_weight_positional*/ false,
                /*use_offsets*/ false);
          } else {
            CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
            kernel64_ = fbgemm::GenerateEmbeddingSpMDMNBit<std::int64_t>(
                BIT_RATE,
                block_size,
                weights != nullptr,
                is_mean,
                /*prefetch distance*/ 8,
                /*is_weight_positional*/ false,
                /*use_offsets*/ false);
          }
        }

        bool success;
        if (std::is_same<IndexType, std::int32_t>::value) {
          success = kernel32_(
              output_size,
              index_size,
              data_size,
              input_data,
              reinterpret_cast<const std::int32_t*>(indices_data),
              lengths_data,
              weights,
              output_data);
        } else {
          success = kernel64_(
              output_size,
              index_size,
              data_size,
              input_data,
              reinterpret_cast<const std::int64_t*>(indices_data),
              lengths_data,
              weights,
              output_data);
        }

        if (success) {
          return true;
        }

        // Error handling
        int64_t current = 0;
        for (int m = 0; m < output_size; ++m) {
          for (int i = 0; i < lengths_data[m]; ++i) {
            CAFFE_ENFORCE_LT(current, index_size);
            IndexType idx = indices_data[current];
            CAFFE_ENFORCE(
                0 <= idx && idx < data_size,
                "Index ",
                current,
                " is out of bounds: ",
                idx,
                ", range 0 to ",
                data_size);
            ++current;
          }
        }
        CAFFE_ENFORCE_EQ(
            current,
            index_size,
            "Your input seems to be incorrect: the sum of lengths values should be "
            "the size of the indices tensor, but it appears not.");

        return false;
    #else
        C10_LOG_EVERY_N(WARNING, 10)
            << "Running slow path because FBGEMM is not available";

        int64_t current = 0;
        for (int m = 0; m < output_size; ++m) {
          memset(output_data, 0, block_size * sizeof(float));
          if (current + lengths_data[m] > index_size) {
            return false;
          }
          for (int i = 0; i < lengths_data[m]; ++i, ++current) {
            IndexType idx = indices_data[current];
            if (idx < 0 || idx >= data_size) {
              return false;
            }

            const at::Half* scale_bias = reinterpret_cast<const at::Half*>(
                input_data + (idx + 1) * data.size(1) - 2 * sizeof(at::Half));

            float weight = 1.0f;
            if (with_weights) {
              weight = weights[current];
            }
            const float scale = weight * scale_bias[0];
            const float bias = weight * scale_bias[1];

            for (int j = 0; j < block_size; ++j) {
              uint8_t quantized =
                  input_data[idx * data.size(1) + j / NUM_ELEM_PER_BYTE];
              quantized >>= (j % NUM_ELEM_PER_BYTE) * BIT_RATE;
              quantized &= (1 << BIT_RATE) - 1;

              output_data[j] = std::fma(scale, quantized, output_data[j] + bias);
            }
          } // for each i
          if (is_mean && lengths_data[m]) {
            float scale = 1.0f / lengths_data[m];
            for (int j = 0; j < block_size; ++j) {
              output_data[j] *= scale;
            }
          }
          output_data += block_size;
        } // for each m

        return current == index_size;
    #endif // USE_FBGEMM
        */
    }
}

