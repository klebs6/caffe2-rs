crate::ix!();

impl BooleanMaskLengthsOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(0);
      auto& mask = Input(1);
      auto* dataOut = Output(0);
      CAFFE_ENFORCE(data.dim() >= 1);
      CAFFE_ENFORCE_EQ(mask.dim(), 1);
      CAFFE_ENFORCE(data.size(0) == mask.size(0));

      const auto* maskPtr = mask.template data<bool>();
      int numOutputs = 0;
      int outerSize = mask.numel();
      for (int i = 0; i < outerSize; ++i) {
        if (maskPtr[i]) {
          ++numOutputs;
        }
      }
      std::vector<int64_t> outShape;
      outShape.push_back(numOutputs);
      outShape.insert(outShape.end(), data.sizes().begin() + 1, data.sizes().end());
      dataOut->Resize(outShape);
      auto* outPtr = (char*)dataOut->raw_mutable_data(data.dtype());

      int64_t* out_vec = nullptr;
      if (OutputSize() == 2) {
        auto* indicesOut = Output(1, {numOutputs}, at::dtype<int64_t>());
        out_vec = indicesOut->template mutable_data<int64_t>();
      }

      if (numOutputs == 0) {
        return true;
      }
      const auto innerSize = data.size_from_dim(1);
      const auto innerSizeBytes = innerSize * data.dtype().itemsize();

      int64_t lastStart = -1;
      const auto* inPtr = (char*)data.raw_data();
      int64_t outStart = 0;

      for (int64_t i = 0;; ++i) {
        // mask was true and either a) became false, or b) sequence finished
        if (lastStart != -1 && ((i >= outerSize) || !maskPtr[i])) {
          const auto* src = inPtr + lastStart * innerSizeBytes;
          auto* dst = outPtr + outStart * innerSizeBytes;
          int numItems = i - lastStart;
          context_.CopyItemsSameDevice(
              data.dtype(), numItems * innerSize, src, dst);
          outStart += numItems;
          lastStart = -1;
        }
        if (i >= outerSize) {
          break;
        }
        // mask was false and became true
        if (lastStart == -1 && maskPtr[i]) {
          lastStart = i;
        }
        if (maskPtr[i] && OutputSize() == 2) {
          *(out_vec++) = i;
        }
      }
      return true;
        */
    }
}

register_cpu_operator!{
    BooleanMask, 
    BooleanMaskOp<CPUContext>
}
