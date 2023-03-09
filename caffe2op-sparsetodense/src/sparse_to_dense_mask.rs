crate::ix!();

/**
  | Convert sparse representations to
  | dense with given indices.
  | 
  | Transforms a sparse representation
  | of map<id, value> represented as `indices`
  | vector and `values` tensor into a compacted
  | tensor where the first dimension corresponds
  | to each id provided in the mask argument.
  | Missing values are filled with the value
  | of `default_value`. After running
  | this op:
  | 
  | output[j, :] = values[i] // where mask[j]
  | == indices[i]
  | 
  | output[j, ...] = default_value //
  | when mask[j] doesn't appear in indices
  | 
  | If `lengths` is provided and not empty,
  | an extra "batch" dimension is prepended
  | to the output.
  | 
  | `values` and `default_value` can have
  | additional matching dimensions (the
  | operation is performed on the entire
  | subtensor in this case).
  | 
  | For example, if `lengths` is supplied
  | and `values` is a 1-D vector of floats
  | and `default_value` is a float scalar,
  | the output is going to be a float matrix
  | of size `len(lengths) X len(mask)`.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseToDenseMaskOp<Context> {
    base:                  SparseToDenseMaskBase<Context>,
    return_presence_mask:  bool,
    max_skipped_rows:      u32, // default = 0
    skipped_rows:          u32, // default = 0
}

impl<Context> SparseToDenseMaskOp<Context> {

    const kMaxSkippedSparseIndices: u32 = 50;

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : SparseToDenseMaskBase<Context>(std::forward<Args>(args)...) 

        returnPresenceMask_ =
            this->template GetSingleArgument<bool>("return_presence_mask", false);
        maxSkippedRows_ = this->template GetSingleArgument<int32_t>(
            "max_skipped_indices", kMaxSkippedSparseIndices);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<TInd>(&mut self) -> bool {
    
        todo!();
        /*
            auto& sparse_indices = Input(INDICES);
        CAFFE_ENFORCE_EQ(sparse_indices.dim(), 1);
        auto& sparse_values = Input(VALUES);
        CAFFE_ENFORCE_GE(sparse_values.dim(), 1);
        CAFFE_ENFORCE_EQ(sparse_indices.numel(), sparse_values.size(0));
        auto& default_value = Input(DEFAULT);
        CAFFE_ENFORCE_EQ(default_value.dim() + 1, sparse_values.dim());
        CAFFE_ENFORCE_EQ(default_value.numel(), sparse_values.size_from_dim(1));
        CAFFE_ENFORCE(sparse_values.dtype() == default_value.dtype());

        const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
        const char* sparse_values_vec =
            static_cast<const char*>(sparse_values.raw_data());
        const void* default_val = default_value.raw_data();

        int64_t block_size = default_value.numel();
        size_t block_nbytes = default_value.nbytes();

        const size_t cols = this->featuresCount_;
        int rows = -1;
        int32_t sparse_indices_length = sparse_indices.dim32(0);
        const int32_t* lengths_vec = nullptr;
        auto* output = Output(OUTPUTVALUE);
        Tensor* presence_mask = nullptr;
        if (returnPresenceMask_) {
          presence_mask = Output(PRESENCEMASK);
        }
        vector<int64_t> shape;
        if (InputSize() == 4) {
          auto& lengths = Input(LENGTHS);
          CAFFE_ENFORCE_EQ(lengths.dim(), 1);
          lengths_vec = lengths.template data<int32_t>();
          rows = lengths.dim32(0);
        }
        if (rows == -1) {
          // if the LENGTHS is not set, the output will be a vector
          rows = 1;
          lengths_vec = &sparse_indices_length;
        } else {
          shape.push_back(rows);
        }
        shape.push_back(cols);
        if (returnPresenceMask_) {
          presence_mask->Resize(shape);
        }
        shape.insert(
            shape.end(),
            default_value.sizes().begin(),
            default_value.sizes().end());
        output->Resize(shape);

        // init
        // TODO: consider unrolling CopyItems to make elemental types copy faster
        char* output_data =
            static_cast<char*>(output->raw_mutable_data(sparse_values.dtype()));
        for (int i = 0; i < cols * rows; i++) {
          context_.CopyItemsSameDevice(
              default_value.dtype(),
              block_size,
              default_val,
              output_data + i * block_nbytes);
        }
        bool* presence_mask_data = nullptr;
        if (returnPresenceMask_) {
          presence_mask_data = presence_mask->template mutable_data<bool>();
          math::Set<bool, Context>(
              rows * cols, false, presence_mask_data, &context_);
        }

        int64_t offset = 0;
        for (int r = 0; r < rows; r++) {
          bool skippedSparseIndex = false;
          for (int c = 0; c < lengths_vec[r]; c++) {
            const auto sparse_index = sparse_indices_vec[offset + c];
            if (sparse_index < 0 ||
                sparse_index >= TInd::max) {
              skippedSparseIndex = true;
              LOG(WARNING) << "Skipping invalid sparse index: " << sparse_index;
              continue;
            }
            int idx = this->getFeatureIdx(sparse_index);
            if (idx != -1) {
              context_.CopyItemsSameDevice(
                  sparse_values.dtype(),
                  block_size,
                  sparse_values_vec + (offset + c) * block_nbytes,
                  output_data + (r * cols + idx) * block_nbytes);
              if (returnPresenceMask_) {
                presence_mask_data[r * cols + idx] = true;
              }
            }
          }
          skippedRows_ += skippedSparseIndex;
          CAFFE_ENFORCE_LT(
              skippedRows_,
              maxSkippedRows_,
              "Too many rows with invalid sparse indices skipped");
          offset += lengths_vec[r];
        }

        return true;
        */
    }
}
