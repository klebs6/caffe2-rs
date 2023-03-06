crate::ix!();

impl<Context> LastNWindowCollectorOp<Context> {
    
    #[inline] pub fn collect(&mut self) -> bool {
        
        todo!();
        /*
        auto* output = Output(LAST_N);
        const auto& input = Input(DATA);

        CAFFE_ENFORCE_GE(input.dim(), 1);
        bool output_initialized = output->numel() > 0 &&
            (static_cast<std::shared_ptr<std::vector<TensorCPU>>*>(
                 output->raw_mutable_data(input.dtype()))[0] != nullptr);
        if (output_initialized) {
          CAFFE_ENFORCE_EQ(output->dim(), input.dim());
          for (size_t i = 1; i < input.dim(); ++i) {
            CAFFE_ENFORCE_EQ(output->size(i), input.size(i));
          }
        }

        auto num_entries = input.sizes()[0];

        if (OutputSize() > NUM_VISITED) {
          auto* num_visited_tensor = Output(NUM_VISITED);
          CAFFE_ENFORCE_EQ(1, num_visited_tensor->numel());
          auto* num_visited = num_visited_tensor->template mutable_data<int64_t>();
          if (!output_initialized) {
            *num_visited = 0;
          }
          CAFFE_ENFORCE_GE(*num_visited, 0);
          *num_visited += num_entries;
        }

        if (!output_initialized) {
          auto dims = input.sizes().vec();
          dims[0] = 0;
          output->Resize(dims);
          // pass meta to output
          output->raw_mutable_data(input.dtype());
          output->ReserveSpace(numToCollect_);
        }

        if (num_entries == 0) {
          if (!output_initialized) {
            // Get both shape and meta
            output->CopyFrom(input, true /*async*/);
          }
          return true;
        }

        auto num_to_copy = std::min<int32_t>(num_entries, numToCollect_);
        auto output_batch_size = output_initialized ? output->size(0) : 0;
        auto output_num =
            std::min<size_t>(numToCollect_, output_batch_size + num_to_copy);

        // output_num is >= output_batch_size
        if (output_num > output_batch_size) {
          output->ExtendTo(output_num, 50);
        }

        auto* output_data =
            static_cast<char*>(output->raw_mutable_data(input.dtype()));

        auto* next = Output(NEXT);
        CAFFE_ENFORCE_EQ(0, next->dim());
        auto* next_data = next->template mutable_data<int32_t>();
        if (!output_initialized) {
          *next_data = 0;
        }
        CAFFE_ENFORCE_LT(*next_data, output->size(0));

        auto block_size = input.size_from_dim(1);
        auto block_bytesize = block_size * input.itemsize();
        const auto* input_data = static_cast<const char*>(input.raw_data());

        if (num_entries > numToCollect_) {
          // just copy the last N rows
          context_.CopyItemsSameDevice(
              input.dtype(),
              num_to_copy * block_size,
              input_data + (num_entries - numToCollect_) * block_bytesize,
              output_data);
          *next_data = 0;
          return true;
        }
        auto start = *next_data;
        auto first_chunk_size =
            std::min<size_t>(num_to_copy + start, numToCollect_) - start;
        context_.CopyItemsSameDevice(
            input.dtype(),
            first_chunk_size * block_size,
            input_data,
            output_data + start * block_bytesize);

        context_.CopyItemsSameDevice(
            input.dtype(),
            (num_to_copy - first_chunk_size) * block_size,
            input_data + first_chunk_size * block_bytesize,
            output_data);

        *next_data = (start + num_to_copy) % numToCollect_;

        return true;
        */
    }
}
