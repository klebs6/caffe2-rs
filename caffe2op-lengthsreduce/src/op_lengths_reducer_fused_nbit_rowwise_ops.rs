crate::ix!();

use crate::{
    OperatorStorage,
    Workspace,
    OperatorDef,
    CPUContext
};

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

pub struct SparseLengthsFusedNBitRowwiseOp<const BIT_RATE: i32,Context,const with_weights: bool,const is_mean: bool> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
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

///-------------------------
pub struct SparseLengthsSumSparseLookupOp {
    storage: OperatorStorage,
    context: CPUContext,
}

enum SparseLengthsSumSparseLookupOpTags {
    INDICES = 0,
    LENGTHS = 1,
    COMPRESSED_INDICES_MAPPING = 2,
    WEIGHTS = 3
}

impl SparseLengthsSumSparseLookupOp {

    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(def, ws)
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
            const auto& indices = Input(INDICES);
        const auto& lengths = Input(LENGTHS);
        const auto& compressed_indices_mapping = Input(COMPRESSED_INDICES_MAPPING);
        thread_local static std::vector<float> dummy_weight;
        CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES must be a vector");
        CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be a vector");
        CAFFE_ENFORCE_EQ(
            compressed_indices_mapping.dim(), 1, "LENGTHS must be a vector");
        const int32_t* lengths_data = lengths.template data<int32_t>();
        const IndexType* indices_data = indices.template data<IndexType>();
        const int32_t* compressed_indices_mapping_data =
            compressed_indices_mapping.template data<std::int32_t>();
        dummy_weight.resize(indices.size(0));
        const float* weights = dummy_weight.data();
        bool has_weights = (InputSize() > 3);
        if (has_weights) {
          const auto& weights_input = Input(WEIGHTS);
          CAFFE_ENFORCE_EQ(weights_input.dim(), 1, "WEIGHTS must be a vector");
          CAFFE_ENFORCE_EQ(
              weights_input.numel(),
              indices.numel(),
              "WEIGHTS should have the same length as INDICES.");
          weights = weights_input.template data<float>();
        }

        // Allocate for the max possible size for now and later we may shrink the
        // indices size.
        auto* output_indices =
            Output(INDICES, indices.sizes(), at::dtype<IndexType>());
        auto* output_lengths =
            Output(LENGTHS, lengths.sizes(), at::dtype<int32_t>());
        Tensor* output_weights = nullptr;
        float* output_weights_data = dummy_weight.data();
        if (has_weights) {
          output_weights = Output(2, indices.sizes(), at::dtype<float>());
          output_weights_data = output_weights->template mutable_data<float>();
        }
        int32_t* output_lengths_data =
            output_lengths->template mutable_data<int32_t>();
        IndexType* output_indices_data =
            output_indices->template mutable_data<IndexType>();
        const int32_t output_size = lengths.size(0);
        const IndexType index_size = indices.size(0);
        const IndexType compressed_data_size = compressed_indices_mapping.size(0);
        IndexType current = 0;
        IndexType current_output = 0;
        for (int m = 0; m < output_size; ++m) {
          const auto current_length = lengths_data[m];
          if (current + current_length > index_size) {
            return false;
          }
          int32_t skipped = 0;
          for (int i = 0; i < current_length; ++i) {
            IndexType compressed_idx = indices_data[current];
            if (compressed_idx < 0 || compressed_idx >= compressed_data_size) {
              return false;
            }
            IndexType idx = compressed_indices_mapping_data[compressed_idx];
            if (idx == -1) {
              ++skipped;
            } else {
              output_weights_data[current_output] = weights[current];
              output_indices_data[current_output++] = idx;
            }
            ++current;
          }
          output_lengths_data[m] = current_length - skipped;
        }

        if (current_output < index_size) {
          output_indices->ShrinkTo(current_output);
          if (output_weights) {
            output_weights->ShrinkTo(current_output);
          }
        }
        return true;
        */
    }
}


///--------------------------------------
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

/**
  | Performs the same operation as SparseLengthsSum,
  | but operating on 4-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsSumFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<4, CPUContext>
}

num_inputs!{SparseLengthsSumFused4BitRowwise, 3}

num_outputs!{SparseLengthsSumFused4BitRowwise, 1}

inputs!{SparseLengthsSumFused4BitRowwise, 
    0 => ("DATA",    "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsSumFused4BitRowwise, 
    0 => ("output",  "output")
}

inherit_onnx_schema!{SparseLengthsSumFused4BitRowwise}

value_key_length_input_fillers!{
    /*
    SparseLengthsSumFused4BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSumFused4BitRowwise}

/**
  | Performs the same operation as
  | SparseLengthsWeightedSum, but operating
  | on 4-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSumFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<4, CPUContext, WithWeights>}

num_inputs!{SparseLengthsWeightedSumFused4BitRowwise, 4}

num_outputs!{SparseLengthsWeightedSumFused4BitRowwise, 1}

inputs!{SparseLengthsWeightedSumFused4BitRowwise, 
    0 => ("DATA",    "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsWeightedSumFused4BitRowwise, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSumFused4BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::LENGTHS,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSumFused4BitRowwise}

/**
  | Performs the same operation as SparseLengthsMean,
  | but operating on 4-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsMeanFused4BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<
        4,
        CPUContext,
        WithoutWeights,
        IsMean>}

num_inputs!{SparseLengthsMeanFused4BitRowwise, 3}

num_outputs!{SparseLengthsMeanFused4BitRowwise, 1}

inputs!{SparseLengthsMeanFused4BitRowwise, 
    0 => ("DATA",    "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsMeanFused4BitRowwise, 
    0 => ("output",  "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMeanFused4BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<4, CPUContext, false, true>::LENGTHS)
    */
}

no_gradient!{SparseLengthsMeanFused4BitRowwise}

/**
  | Performs the same operation as SparseLengthsSum,
  | but operating on 2-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsSumFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<2, CPUContext>}

num_inputs!{SparseLengthsSumFused2BitRowwise, 3}

num_outputs!{SparseLengthsSumFused2BitRowwise, 1}

inputs!{SparseLengthsSumFused2BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsSumFused2BitRowwise, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSumFused2BitRowwise}

value_key_length_input_fillers!{
    /*
    SparseLengthsSumFused2BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSumFused2BitRowwise}

/**
  | Performs the same operation as
  | SparseLengthsWeightedSum, but operating
  | on 2-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSumFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<2, CPUContext, /*with_weights=*/true>}

num_inputs!{SparseLengthsWeightedSumFused2BitRowwise, 4}

num_outputs![SparseLengthsWeightedSumFused2BitRowwise, 1];

inputs!{SparseLengthsWeightedSumFused2BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsWeightedSumFused2BitRowwise, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSumFused2BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::LENGTHS,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSumFused2BitRowwise}

/**
  | Performs the same operation as SparseLengthsMean,
  | but operating on 2-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias).
  |
  */
register_cpu_operator!{
    SparseLengthsMeanFused2BitRowwise,
    SparseLengthsFusedNBitRowwiseOp<
        2,
        CPUContext,
        /*with_weights=*/false,
        /*is_mean=*/true>}

num_inputs!{SparseLengthsMeanFused2BitRowwise, 3}

num_outputs!{SparseLengthsMeanFused2BitRowwise, 1}

inputs!{SparseLengthsMeanFused2BitRowwise, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA")
}

outputs!{SparseLengthsMeanFused2BitRowwise, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMeanFused2BitRowwise, (
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::DATA,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::INDICES,
        SparseLengthsFusedNBitRowwiseOp<2, CPUContext, false, true>::LENGTHS)
    */
}


no_gradient!{SparseLengthsMeanFused2BitRowwise}

/**
  | This op converts compressed indices
  | of SparseLengthsSum*Sparse to uncompressed
  | indices of SparseLengthsSum*.
  | 
  | For compressed indices that maps to
  | -1.
  | 
  | It means it will correspond to a zero
  | row in the uncompressed data.
  | 
  | Therefore we will remove this indices
  | and adjust the lengths.
  |
  */
register_cpu_operator!{
    SparseLengthsSumSparseLookup,
    SparseLengthsSumSparseLookupOp}

num_inputs!{SparseLengthsSumSparseLookup, (3,4)}

num_outputs!{SparseLengthsSumSparseLookup, (2,3)}

inputs!{SparseLengthsSumSparseLookup, 
    0 => ("INDICES", "Integer vector containing compressed indices of the first dimension of DATA for the slices that are being aggregated"),
    1 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of INDICES"),
    2 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices"),
    3 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction. Same size as INDICES.")
}

outputs!{SparseLengthsSumSparseLookup, 
    0 => ("output_indices", "Uncompressed indices"),
    1 => ("output_lengths", "Adjusted lengths"),
    2 => ("output_weights", "Adjusted weights")
}

inherit_onnx_schema!{SparseLengthsSumSparseLookup}

no_gradient!{SparseLengthsSumSparseLookup}

/**
  | Performs SparseLengthsSum, but operating
  | on 4-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and 2-byte fp16 bias), and where
  | rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsSum4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<4>}

num_inputs!{SparseLengthsSum4BitRowwiseSparse, 4}

num_outputs!{SparseLengthsSum4BitRowwiseSparse, 1}

inputs!{SparseLengthsSum4BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsSum4BitRowwiseSparse, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSum4BitRowwiseSparse}

value_key_length_input_fillers!{
    /*
    SparseLengthsSum4BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<4>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSum4BitRowwiseSparse}

/**
  | Performs SparseLengthsWeightedSum,
  | but operating on 4-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias), and
  | where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        4,
        /*with_weights=*/true>}

num_inputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 5}

num_outputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 1}

inputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsWeightedSum4BitRowwiseSparse, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSum4BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<4, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<4, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSum4BitRowwiseSparse}

/**
  | Performs SparseLengthsMean, but operating
  | on 4-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias), and where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsMean4BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        4,
        /*with_weights=*/false,
        /*is_mean=*/true>}

num_inputs!{SparseLengthsMean4BitRowwiseSparse, 4}

num_outputs!{SparseLengthsMean4BitRowwiseSparse, 1}

inputs!{SparseLengthsMean4BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsMean4BitRowwiseSparse, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMean4BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::
            COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<4, false, true>::LENGTHS)
    */
}

no_gradient!{SparseLengthsMean4BitRowwiseSparse}

/**
  | Performs SparseLengthsSum, but operating
  | on 8-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 4-byte fp32
  | scale and 4-byte fp32 bias), and where
  | rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsSum8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<8>}

num_inputs!{SparseLengthsSum8BitRowwiseSparse, 4}

num_outputs!{SparseLengthsSum8BitRowwiseSparse, 1}

inputs!{SparseLengthsSum8BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsSum8BitRowwiseSparse, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSum8BitRowwiseSparse}

value_key_length_input_fillers!{
    /*
    SparseLengthsSum8BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<8>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8>::LENGTHS)
    */
}

no_gradient!{SparseLengthsSum8BitRowwiseSparse}

/**
  | Performs SparseLengthsWeightedSum,
  | but operating on 8-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 4-byte fp32 scale and bias), and
  | where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        8,
        /*with_weights=*/true>}

num_inputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 5}

num_outputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 1}

inputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsWeightedSum8BitRowwiseSparse, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSum8BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<8, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<8, true>::WEIGHTS)
    */
}

no_gradient!{SparseLengthsWeightedSum8BitRowwiseSparse}

/**
  | Performs SparseLengthsMean, but operating
  | on 8-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 4-byte fp32
  | scale and bias), and where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsMean8BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        8,
        /*with_weights=*/false,
        /*is_mean=*/true>}

num_inputs!{SparseLengthsMean8BitRowwiseSparse, 4}

num_outputs!{SparseLengthsMean8BitRowwiseSparse, 1}

inputs!{SparseLengthsMean8BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused4BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsMean8BitRowwiseSparse, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMean8BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::
            COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<8, false, true>::LENGTHS)
        */
}

no_gradient!{SparseLengthsMean8BitRowwiseSparse}

/**
  | Performs SparseLengthsSum, but operating
  | on 2-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and 2-byte fp16 bias), and where
  | rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsSum2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<2>}

num_inputs!{SparseLengthsSum2BitRowwiseSparse, 4}

num_outputs!{SparseLengthsSum2BitRowwiseSparse, 1}

inputs!{SparseLengthsSum2BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsSum2BitRowwiseSparse, 
    0 => ("output", "output")
}

inherit_onnx_schema!{SparseLengthsSum2BitRowwiseSparse}

value_key_length_input_fillers!{
    /*
    SparseLengthsSum2BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<2>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2>::LENGTHS)

        */
}

no_gradient!{SparseLengthsSum2BitRowwiseSparse}

/**
  | Performs SparseLengthsWeightedSum,
  | but operating on 2-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then 2-byte fp16 scale and bias), and
  | where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsWeightedSum2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        2,
        /*with_weights=*/true>
}

num_inputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 5}

num_outputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 1}

inputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("WEIGHTS", "Vector of weights to scale rows of DATA with before reduction"),
    2 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    3 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    4 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsWeightedSum2BitRowwiseSparse, 
    0 => ("output", "output")
}

weighted_value_key_length_input_fillers!{
    /*
    SparseLengthsWeightedSum2BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<2, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2, true>::LENGTHS,
        SparseLengthsNBitRowwiseSparseOp<2, true>::WEIGHTS)

        */
}

no_gradient!{SparseLengthsWeightedSum2BitRowwiseSparse}

/**
  | Performs SparseLengthsMean, but operating
  | on 2-bit rowwise quantized matrices
  | with fused storage (where each row stores
  | quantized values, and then 2-byte fp16
  | scale and bias), and where rows are pruned.
  |
  */
register_cpu_operator!{
    SparseLengthsMean2BitRowwiseSparse,
    SparseLengthsNBitRowwiseSparseOp<
        2,
        /*with_weights=*/false,
        /*is_mean=*/true>
}

num_inputs!{SparseLengthsMean2BitRowwiseSparse, 4}

num_outputs!{SparseLengthsMean2BitRowwiseSparse, 1}

inputs!{SparseLengthsMean2BitRowwiseSparse, 
    0 => ("DATA", "uint8 tensor obtained with operator FloatToFused2BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated"),
    2 => ("LENGTHS", "Vector with the same sum of elements as the first dimension of DATA"),
    3 => ("COMPRESSED_INDICES_MAPPING", "Integer vector mapping uncompressed indices to compressed indices")
}

outputs!{SparseLengthsMean2BitRowwiseSparse, 
    0 => ("output", "output")
}

value_key_length_input_fillers!{
    /*
    SparseLengthsMean2BitRowwiseSparse, (
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::COMPRESSED_INDICES_MAPPING,
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::INDICES,
        SparseLengthsNBitRowwiseSparseOp<2, false, true>::LENGTHS)

        */
}

no_gradient!{SparseLengthsMean2BitRowwiseSparse}

export_caffe2_op_to_c10_cpu!{
    SparseLengthsSum8BitRowwiseSparse,
    "_caffe2::SparseLengthsSum8BitRowwiseSparse(
        Tensor data, 
        Tensor indices, 
        Tensor lengths, 
        Tensor compressed_indices_mapping) -> Tensor output",
        SparseLengthsNBitRowwiseSparseOp::<8>
}
