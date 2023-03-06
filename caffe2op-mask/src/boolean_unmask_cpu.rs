crate::ix!();

impl BooleanUnmaskOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int maskSize = Input(0).numel();
      int numMasks = InputSize() / 2;
      auto& valueMeta = Input(1).dtype();

      auto* valuesOut = Output(0);
      valuesOut->Resize(maskSize);
      auto* valuesOutPtr = (char*)valuesOut->raw_mutable_data(valueMeta);

      std::vector<int> nextValueIndices(numMasks, 0);
      for (int maskOffset = 0; maskOffset < maskSize; ++maskOffset) {
        bool maskFound = false;
        for (int maskIndex = 0; maskIndex < numMasks; ++maskIndex) {
          auto& mask = Input(maskIndex * 2);
          CAFFE_ENFORCE_EQ(mask.dim(), 1);
          CAFFE_ENFORCE_EQ(mask.numel(), maskSize);
          const auto* maskPtr = mask.template data<bool>();

          auto& values = Input(maskIndex * 2 + 1);
          CAFFE_ENFORCE_EQ(values.dim(), 1);
          const auto* valuesPtr = (char*)values.raw_data();

          if (maskPtr[maskOffset]) {
            auto& valueIndex = nextValueIndices[maskIndex];
            CAFFE_ENFORCE_LT(valueIndex, values.numel());
            auto* src = valuesPtr + (valueIndex++) * valueMeta.itemsize();
            auto* dst = valuesOutPtr + maskOffset * valueMeta.itemsize();
            std::copy(src, src + valueMeta.itemsize(), dst);
            maskFound = true;
            break;
          }
        }
        CAFFE_ENFORCE(
            maskFound, "All masks have False at position ", maskOffset, ".");
      }
      // check all indices match value length
      for (int i = 0; i < numMasks; ++i) {
        auto& values = Input(i * 2 + 1);
        CAFFE_ENFORCE_EQ(
            values.numel(),
            nextValueIndices[i],
            "The number of true at mask ",
            i,
            " does not match the corresponding value size.");
      }
      return true;
        */
    }
}
