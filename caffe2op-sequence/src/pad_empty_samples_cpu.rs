crate::ix!();

impl PadEmptySamplesOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& lengths = Input(0);
      auto* lengthsPtr = lengths.template data<int32_t>();
      CAFFE_ENFORCE(lengths.dim() == 1, "LENGTH should be 1-D");
      CAFFE_ENFORCE(InputSize() >= 1, "Input size must be no less than 1");

      int needPadding = 0;
      int sumLen = 0;
      for (int i = 0; i < lengths.numel(); ++i) {
        if (lengthsPtr[i] == 0) {
          needPadding++;
        }
        sumLen += lengthsPtr[i];
      }

      auto* out_lengths = Output(0, {lengths.numel()}, at::dtype<int32_t>());
      auto* outLengthsPtr = out_lengths->template mutable_data<int32_t>();
      for (int i = 0; i < lengths.numel(); ++i) {
        if (lengthsPtr[i] == 0) {
          outLengthsPtr[i] = 1;
        } else {
          outLengthsPtr[i] = lengthsPtr[i];
        }
      }

      for (int k = 0; k < InputSize() - 1; k++) {
        auto& features = Input(1 + k);
        CAFFE_ENFORCE(features.dim() >= 1, "FEATURE should at least 1-D");
        CAFFE_ENFORCE(
            features.size(0) == sumLen, "FEATURE and LENGTH should be consistent");
        const auto block_size = features.size_from_dim(1);

        auto* out_features = Output(1 + k);
        auto outDim = features.sizes().vec();
        outDim.at(0) += needPadding;
        out_features->Resize(outDim);
        auto dst =
            static_cast<char*>(out_features->raw_mutable_data(features.dtype()));
        auto src_base = static_cast<const char*>(features.raw_data());
        // copy data and add padding index as zero
        Tensor zero{CPU};
        zero.Resize(block_size);
        auto zeroPtr = static_cast<char*>(zero.raw_mutable_data(features.dtype()));
        // TODO Handle other composite types, such as vector<...>
        if (!features.dtype().Match<std::string>()) {
          memset(zeroPtr, 0, zero.nbytes());
        }
        int start_dest = 0;
        int start_src = 0;
        for (int i = 0; i < lengths.numel(); ++i) {
          if (lengthsPtr[i] == 0) {
            context_.CopyItemsSameDevice(
                features.dtype(),
                block_size,
                zeroPtr,
                dst + start_dest * features.dtype().itemsize());
            start_dest += block_size;
          } else {
            auto src = src_base + start_src * features.dtype().itemsize();
            context_.CopyItemsSameDevice(
                features.dtype(),
                lengthsPtr[i] * block_size,
                src,
                dst + start_dest * features.dtype().itemsize());
            start_src += lengthsPtr[i] * block_size;
            start_dest += lengthsPtr[i] * block_size;
          }
        }
      }
      return true;
        */
    }
}
