crate::ix!();

impl<Context> ReversePackedSegsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double, int, long, bool>>::call(
            this, Input(DATA));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            if (Input(LENGTHS).template IsType<int>()) {
              DoRunWithLengthType<T, int>();
            } else {
              DoRunWithLengthType<T, long>();
            }
            return true;
        */
    }

    #[inline] pub fn do_run_with_length_type<T, LengthType>(&mut self) {
        todo!();
        /*
            const auto& data = Input(DATA);
            const auto& lengths = Input(LENGTHS);

            CAFFE_ENFORCE(
                data.dim() == 3,
                "DATA should be 3-D tensor <lengths, "
                "segments, embeddings>");
            CAFFE_ENFORCE(lengths.dim() == 1, "LENGTH should be 1-D");

            const auto shape = data.sizes();
            auto* output = Output(0, shape, at::dtype<T>());

            const auto max_length = data.sizes()[0];
            const auto batch_size = data.sizes()[1];
            const auto block_size = data.sizes()[2];
            CAFFE_ENFORCE(
                lengths.sizes()[0] == batch_size,
                "lenths size should be"
                " equal to batch size");

            const T* data_ptr = data.template data<T>();
            const LengthType* lengths_ptr = lengths.template data<LengthType>();

            vector<LengthType> lengths_host(batch_size);
            context_.template CopyToCPU<LengthType>(
                batch_size, lengths_ptr, &lengths_host[0]);
            context_.FinishDeviceComputation();

            T* rev_data_ptr = output->template mutable_data<T>();
            for (int64_t i = 0; i < batch_size; i++) {
              const auto& seg_length = lengths_host[i];
              CAFFE_ENFORCE_LE(seg_length, max_length);
              int64_t j = 0;
              for (; j < seg_length; j++) {
                const T* data_block_ptr = data_ptr + (j * batch_size + i) * block_size;
                T* rev_data_block_ptr =
                    rev_data_ptr + ((seg_length - 1 - j) * batch_size + i) * block_size;
                context_.template CopySameDevice<T>(
                    block_size, data_block_ptr, rev_data_block_ptr);
              }
              for (; j < max_length; j++) {
                const T* data_block_ptr = data_ptr + (j * batch_size + i) * block_size;
                T* rev_data_block_ptr =
                    rev_data_ptr + (j * batch_size + i) * block_size;
                context_.template CopySameDevice<T>(
                    block_size, data_block_ptr, rev_data_block_ptr);
              }
            }
        */
    }
}
