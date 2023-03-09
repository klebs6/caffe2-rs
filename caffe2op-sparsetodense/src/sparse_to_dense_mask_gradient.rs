crate::ix!();

/**
  | The output is the gradient of the input
  | value from SparseToDenseMask.
  | 
  | The gradient for default_value has
  | not been implemented.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SparseToDenseMaskGradientOp<Context> {
    base: SparseToDenseMaskBase<Context>,
}

impl<Context> SparseToDenseMaskGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : SparseToDenseMaskBase<Context>(std::forward<Args>(args)...)
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
        auto& gradient_output = Input(GOUTPUT);

        int64_t block_size = gradient_output.size_from_dim(1);
        size_t block_nbytes = gradient_output.itemsize() * block_size;

        const size_t cols = this->featuresCount_;
        int rows = -1;
        int iter_offset = 1;
        int32_t default_length = sparse_indices.dim32(0);
        const int32_t* lengths_vec = nullptr;
        auto* output = Output(GVALUES);
        vector<int64_t> shape;
        if (InputSize() > LENGTHS) {
          // if the LENGTHS is set, the gradient_output has dim:
          // lengths * mask.size() * feature_dim
          auto& lengths = Input(LENGTHS);
          lengths_vec = lengths.template data<int32_t>();
          rows = lengths.dim32(0);
          CAFFE_ENFORCE_EQ(lengths.dim(), 1);
          CAFFE_ENFORCE_GE(gradient_output.dim(), 2);
          CAFFE_ENFORCE_EQ(gradient_output.size(0), rows);
          CAFFE_ENFORCE_EQ(gradient_output.size(1), cols);
          block_nbytes /= gradient_output.size(1);
          block_size /= gradient_output.size(1);
          iter_offset += 1;
        }
        if (rows == -1) {
          // if the LENGTHS is not set, the gradient_output has dim:
          // mask.size() * feature_dim
          rows = 1;
          lengths_vec = &default_length;
          CAFFE_ENFORCE_GE(gradient_output.dim(), 1);
          CAFFE_ENFORCE_EQ(gradient_output.size(0), cols);
        }
        shape.push_back(default_length);
        // insert feature_dim
        shape.insert(
            shape.end(),
            gradient_output.sizes().begin() + iter_offset,
            gradient_output.sizes().end());
        output->Resize(shape);

        const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
        const char* gradient_output_vec =
            static_cast<const char*>(gradient_output.raw_data());

        char* output_data =
            static_cast<char*>(output->raw_mutable_data(gradient_output.dtype()));
        memset(output_data, 0, output->nbytes());
        math::Set<char, Context>(
            default_length * gradient_output.itemsize(), 0, output_data, &context_);

        int32_t offset = 0;
        // SparseToDenseMask is not injective; gradient_used records
        // if the gradient is used for other input value from the same row
        vector<bool> gradient_used(cols, false);
        for (int r = 0; r < rows; r++) {
          std::fill(gradient_used.begin(), gradient_used.end(), false);
          for (int c = lengths_vec[r] - 1; c >= 0; c--) {
            int idx = this->getFeatureIdx(sparse_indices_vec[offset + c]);
            if (idx != -1 && !gradient_used[idx]) {
              gradient_used[idx] = true;
              context_.CopyItemsSameDevice(
                  gradient_output.dtype(),
                  block_size,
                  gradient_output_vec + (r * cols + idx) * block_nbytes,
                  output_data + (offset + c) * block_nbytes);
            }
          }
          offset += lengths_vec[r];
        }
        return true;
        */
    }
}
