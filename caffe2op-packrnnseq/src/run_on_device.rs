crate::ix!();

impl<Context,const Forward: bool> 
PackRNNSequenceOpBase<Context,Forward> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t, float, double>>::call(
            this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<ValT>(&mut self) -> bool {
        todo!();
        /*
            // The value is copied from the sequence to the pack
        // if Forward is true, and vice versa
        int dim_offset = Forward ? 1 : 2;
        auto& values = Input(0);
        CAFFE_ENFORCE_GT(values.dim(), dim_offset);

        // block_size is the size for each individual feature
        int64_t block_size = values.size_from_dim(dim_offset);
        auto values_vec = values.template data<ValT>();

        auto& lengths = Input(LENGTHS);
        CAFFE_ENFORCE_EQ(lengths.dim(), 1);
        const auto cols = lengths.numel();
        const int32_t* lengths_vec = lengths.template data<int32_t>();
        // the total number of rows is defined as the max number from lengths
        // if when the lengths is empty, we set rows = 0 to support zero lengths
        const auto rows =
            cols ? *std::max_element(lengths_vec, lengths_vec + cols) : 0;
        CAFFE_ENFORCE_GE(rows, 0);
        int length_sum = 0;
        if (cols > 0) {
          math::Sum<int, Context>(cols, lengths_vec, &length_sum, &context_);
        }

        vector<int64_t> shape;
        // the output shape is rows * cols for the pack,
        // or length_sum for the sequence
        if (Forward) {
          shape.push_back(rows);
          shape.push_back(cols);
        } else {
          shape.push_back(length_sum);
        }
        // insert the dim for the feature
        shape.insert(
            shape.end(), values.sizes().begin() + dim_offset, values.sizes().end());

        auto* output = Output(OUTPUTVALUE, shape, at::dtype<ValT>());

        auto output_data = output->template mutable_data<ValT>();
        // initialize output_data with zero, as it is the default value for padding
        // when certain length is smaller than rows
        math::Set<ValT, Context>(output->numel(), 0, output_data, &context_);

        int32_t offset = 0;
        for (int c = 0; c < cols; c++) {
          for (int r = 0; r < lengths_vec[c]; r++) {
            auto input_offset = Forward ? (offset + r) : (r * cols + c);
            auto output_offset = Forward ? (r * cols + c) : (offset + r);
            context_.CopyItemsSameDevice(
                values.dtype(),
                block_size,
                values_vec + input_offset * block_size,
                output_data + output_offset * block_size);
          }
          offset += lengths_vec[c];
        }
        return true;
        */
    }
}
