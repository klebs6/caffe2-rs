crate::ix!();

impl<Context> GatherRangesOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(RANGES, CPU));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
        auto& ranges = Input(RANGES);
        auto* outputData = Output(0);
        auto* outputLengths = Output(1);

        auto batchSize = ranges.size(0);
        CAFFE_ENFORCE(data.dim() == 1, "Data has to be 1-D");
        CAFFE_ENFORCE(ranges.dim() == 3, "Ranges must be 3-D");
        CAFFE_ENFORCE(ranges.size(1) > 0, "There has to be at least one range");
        CAFFE_ENFORCE_EQ(
            ranges.size(2), 2, "Ranges last dimension should be of size 2");

        auto* rawData = static_cast<const char*>(data.raw_data());
        auto* rangesData = ranges.template data<Index>();

        outputLengths->Resize(batchSize);
        auto* outputLengthsPtr = outputLengths->template mutable_data<int32_t>();
        size_t start = 0;
        size_t blockSize = ranges.size_from_dim(1);
        for (size_t i = 0; i < batchSize; ++i) {
          auto end = start + blockSize;
          outputLengthsPtr[i] = accumulate(rangesData, start, end);
          start = end;
        }

        size_t outputSize = accumulate(rangesData, 0, ranges.numel());
        outputData->Resize(outputSize);

        auto outputRawData =
            static_cast<char*>(outputData->raw_mutable_data(data.dtype()));
        VLOG(1) << "Copying data";
        size_t outputOffsetBytes = 0;
        auto itemsize = data.dtype().itemsize();
        for (int i = 0; i < ranges.numel(); i += 2) {
          auto rangeStart = rangesData[i];
          auto rangeLength = rangesData[i + 1];
          if (!rangeLength) {
            continue;
          }
          auto rangeSizeBytes = rangeLength * itemsize;
          CAFFE_ENFORCE(outputOffsetBytes < outputSize * itemsize);
          CAFFE_ENFORCE(rangeStart + rangeLength <= data.numel());
          context_.CopyItemsSameDevice(
              data.dtype(),
              rangeLength,
              rawData + rangeStart * itemsize,
              outputRawData + outputOffsetBytes);
          outputOffsetBytes += rangeSizeBytes;
        }
        CAFFE_ENFORCE(outputOffsetBytes == outputSize * itemsize);
        return true;
        */
    }
    
    #[inline] pub fn accumulate<Index>(
        &mut self, 
        ranges: *mut Index,
        start:  usize,
        end:    usize) -> usize 
    {
        todo!();
        /*
            size_t result = 0;
        for (size_t i = start + 1; i < end; i += 2) {
          result += ranges[i];
        }
        return result;
        */
    }
}
