crate::ix!();

pub struct SparseLengthsSumSparseLookupOp {
    storage: OperatorStorage,
    context: CPUContext,
}

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
