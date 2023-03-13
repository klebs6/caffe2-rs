crate::ix!();

impl<Context> LengthsGatherOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(INDICES, CPU));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& items = Input(ITEMS);
        auto& lengths = Input(LENGTHS);
        auto& indices = Input(INDICES);
        auto* output = Output(0);

        CAFFE_ENFORCE_GE(items.dim(), 1, "ITEMS should be at least 1-D");
        CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS should be 1-D");
        CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES should be 1-D");

        const auto* lengths_data = lengths.template data<int32_t>();
        const auto* indices_data = indices.template data<Index>();

        int64_t total_length = 0;
        for (size_t i = 0; i < indices.numel(); ++i) {
          auto idx = indices_data[i];
          CAFFE_ENFORCE_LT(idx, lengths.numel());
          total_length += lengths_data[idx];
        }
        auto shape = items.sizes().vec();
        shape[0] = total_length;
        output->Resize(shape);

        offsets_.clear();
        int64_t running_offset = 0;
        offsets_.reserve(lengths.numel());
        for (size_t i = 0; i < lengths.numel(); ++i) {
          offsets_.push_back(running_offset);
          running_offset += lengths_data[i];
        }
        CAFFE_ENFORCE_EQ(
            items.size(0),
            running_offset,
            "LENGTHS must match the first dimension of ITEMS");

        auto src_base = static_cast<const char*>(items.raw_data());
        auto block_size = items.size_from_dim(1);
        auto block_bytesize = block_size * items.itemsize();
        auto out = static_cast<char*>(output->raw_mutable_data(items.dtype()));

        for (size_t i = 0; i < indices.numel(); ++i) {
          auto idx = indices_data[i];
          auto length = lengths_data[idx];
          context_.CopyItemsSameDevice(
              items.dtype(),
              length * block_size,
              src_base + offsets_[idx] * block_bytesize,
              out);
          out += length * block_bytesize;
        }
        return true;
        */
    }
}

