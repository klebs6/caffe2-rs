crate::ix!();

pub struct SparseLengthsNBitRowwiseSparseOp<const BIT_RATE: i32,const with_weights: bool,const is_mean: bool> {
    storage: OperatorStorage,
    context: CPUContext,

    /*
    #ifdef USE_FBGEMM
     private:
      std::int64_t last_block_size{-1};
      fbgemm::EmbeddingSpMDMRowWiseSparseKernelSignature<
          std::uint8_t,
          std::int32_t>::Type kernel32_;
      fbgemm::EmbeddingSpMDMRowWiseSparseKernelSignature<
          std::uint8_t,
          std::int64_t>::Type kernel64_;
      fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int32_t>::Type
          kernel32_no_sparse_;
      fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int64_t>::Type
          kernel64_no_sparse_;
    #endif
    */
}

/**
  |note there is something weird here because we
  |may or may not have a Weights input therefore
  |check how we use Input(N) where N is integer for
  |correct usage (see c++ if this is confusing)
  */
pub enum SparseLengthsNBitRowwiseSparseOpTags {
    Data,
    MaybeWeights,
    Indices,
    Lengths,
    CompressedIndicesMapping,
}

impl<const BIT_RATE: i32,const with_weights: bool,const is_mean: bool> 
SparseLengthsNBitRowwiseSparseOp<BIT_RATE,with_weights,is_mean> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
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
        const auto& compressed_indices_mapping = Input(COMPRESSED_INDICES_MAPPING);
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
        // Subtract 4 (or 8 for BIT_RATE == 8) from the #columns of data for the
        // fp16 (or fp32 for BIT_RATE == 8) scale and bias that we use in the fused
        // representation (per row).
        const std::vector<int64_t> shape = {
            lengths.size(0),
            static_cast<int64_t>(
                data.size(1) -
                2 * (BIT_RATE == 8 ? sizeof(float) : sizeof(at::Half))) *
                NUM_ELEM_PER_BYTE};
        auto* output = Output(0, shape, at::dtype<float>());

        int output_size = output->size(0);
        int block_size = output->size(1);
        CAFFE_ENFORCE_EQ(
            block_size % NUM_ELEM_PER_BYTE,
            0,
            "block size must be divisible by " + std::to_string(NUM_ELEM_PER_BYTE));
        auto data_size = data.size(0);
        int index_size = indices.numel();
        const uint8_t* input_data = data.template data<uint8_t>();
        const IndexType* indices_data = indices.template data<IndexType>();
        const int* lengths_data = lengths.template data<int>();
        float* output_data = output->template mutable_data<float>();
        const std::int32_t* compressed_indices_mapping_data =
            compressed_indices_mapping.template data<std::int32_t>();

        // if compressed_indices_mapping is [0], it is a indicator that
        // we should fallback to normal SLS, which is also a valid fallback if
        // the LUT is pruned.
        const bool fallback_to_no_sparse =
            (compressed_indices_mapping.numel() == 1 &&
             compressed_indices_mapping_data[0] == 0);

    #ifdef USE_FBGEMM
        // If this is the first call or block size has changed (should never happen
        // actually), generate a kernel.
        if (block_size != last_block_size) {
          if (!fallback_to_no_sparse) {
            last_block_size = block_size;
            if (std::is_same<IndexType, std::int32_t>::value) {
              if (BIT_RATE == 8) {
                kernel32_ = fbgemm::
                    GenerateEmbeddingSpMDMRowWiseSparse<std::uint8_t, std::int32_t>(
                        block_size,
                        weights != nullptr,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              } else {
                kernel32_ =
                    fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<std::int32_t>(
                        BIT_RATE,
                        block_size,
                        weights != nullptr,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              }
            } else {
              CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
              if (BIT_RATE == 8) {
                kernel64_ = fbgemm::
                    GenerateEmbeddingSpMDMRowWiseSparse<std::uint8_t, std::int64_t>(
                        block_size,
                        weights != nullptr,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              } else {
                kernel64_ =
                    fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<std::int64_t>(
                        BIT_RATE,
                        block_size,
                        weights != nullptr,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              }
            }
          } else { // fallback_to_no_sparse == true
            last_block_size = block_size;
            if (std::is_same<IndexType, std::int32_t>::value) {
              if (BIT_RATE == 8) {
                kernel32_no_sparse_ =
                    fbgemm::GenerateEmbeddingSpMDM<std::uint8_t, std::int32_t>(
                        block_size,
                        with_weights,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              } else {
                kernel32_no_sparse_ =
                    fbgemm::GenerateEmbeddingSpMDMNBit<std::int32_t>(
                        BIT_RATE,
                        block_size,
                        weights != nullptr,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              }
            } else {
              CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
              if (BIT_RATE == 8) {
                kernel64_no_sparse_ =
                    fbgemm::GenerateEmbeddingSpMDM<std::uint8_t, std::int64_t>(
                        block_size,
                        with_weights,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              } else {
                kernel64_no_sparse_ =
                    fbgemm::GenerateEmbeddingSpMDMNBit<std::int64_t>(
                        BIT_RATE,
                        block_size,
                        weights != nullptr,
                        is_mean,
                        /*prefetch distance*/ 16,
                        /*is_weight_positional*/ false,
                        /*use_offsets*/ false);
              }
            }
          }
        } // end if (block_size != last_block_size)

        bool success;
        if (!fallback_to_no_sparse) {
          if (std::is_same<IndexType, std::int32_t>::value) {
            success = kernel32_(
                output_size,
                index_size,
                compressed_indices_mapping.size(),
                input_data,
                reinterpret_cast<const std::int32_t*>(indices_data),
                lengths_data,
                weights,
                output_data,
                compressed_indices_mapping_data);
          } else {
            success = kernel64_(
                output_size,
                index_size,
                compressed_indices_mapping.size(),
                input_data,
                reinterpret_cast<const std::int64_t*>(indices_data),
                lengths_data,
                weights,
                output_data,
                compressed_indices_mapping_data);
          }
        } else { // fallback_to_no_sparse == true
          if (std::is_same<IndexType, std::int32_t>::value) {
            success = kernel32_no_sparse_(
                output_size,
                index_size,
                data_size,
                input_data,
                reinterpret_cast<const std::int32_t*>(indices_data),
                lengths_data,
                weights,
                output_data);
          } else {
            success = kernel64_no_sparse_(
                output_size,
                index_size,
                data_size,
                input_data,
                reinterpret_cast<const std::int64_t*>(indices_data),
                lengths_data,
                weights,
                output_data);
          }
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
            if (!fallback_to_no_sparse) {
              CAFFE_ENFORCE(
                  0 <= idx && idx < compressed_indices_mapping.size(),
                  "Index ",
                  current,
                  " is out of bounds: ",
                  idx,
                  ", range 0 to ",
                  compressed_indices_mapping.size());
            } else {
              CAFFE_ENFORCE(
                  0 <= idx && idx < data_size,
                  "Index ",
                  current,
                  " is out of bounds: ",
                  idx,
                  ", range 0 to ",
                  data_size);
            }
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
            IndexType idx;
            if (fallback_to_no_sparse) {
              idx = indices_data[current];
              if (idx < 0 || idx >= data_size) {
                return false;
              }
            } else {
              IndexType uncompressed_idx = indices_data[current];
              if (uncompressed_idx < 0 ||
                  uncompressed_idx >= compressed_indices_mapping.size()) {
                return false;
              }
              idx = compressed_indices_mapping_data[uncompressed_idx];
              if (idx == -1) {
                continue;
              }
            }

            const uint8_t* scale_bias = input_data + (idx + 1) * data.size(1) -
                2 * (BIT_RATE == 8 ? sizeof(float) : sizeof(at::Half));

            float weight = 1.0f;
            if (with_weights) {
              weight = weights[current];
            }
            float scale, bias;
            if (BIT_RATE == 8) {
              scale = weight * reinterpret_cast<const float*>(scale_bias)[0];
              bias = weight * reinterpret_cast<const float*>(scale_bias)[1];
            } else {
              scale = weight * reinterpret_cast<const at::Half*>(scale_bias)[0];
              bias = weight * reinterpret_cast<const at::Half*>(scale_bias)[1];
            }

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
