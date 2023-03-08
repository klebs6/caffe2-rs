crate::ix!();

/**
  | Inverse operation of Partition.
  | 
  | Takes the original, full 'keys' tensor
  | followed by sharded value tensors,
  | and returns the full value tensor, combined
  | using the same hash used in
  | Partition.
  |
  */
#[USE_DISPATCH_HELPER]
#[USE_OPERATOR_FUNCTIONS("CPUContext")]
pub struct GatherByKeyOp {
    storage:          OperatorStorage,
    context:          CPUContext,
    input_datas:      Vec<*mut u8>,
    in_start_offsets: Vec<i64>,
}

num_inputs!{GatherByKey, (2,INT_MAX)}

num_outputs!{GatherByKey, 1}

inputs!{
    GatherByKey, 
    0 => ("keys", "The first input is the full keys tensor (same as the first input of Partition)."),
    1 => ("sharded_values", "Subsequented inputs are sharded values tensors.")
}

outputs!{
    GatherByKey, 
    0 => ("values", "Reconstructed values tensor.")
}

impl GatherByKeyOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            const auto numPartitions = InputSize() - 1;
        CAFFE_ENFORCE_GE(numPartitions, 1);
        const auto& keysTensor = Input(0);
        const auto* keysData = keysTensor.template data<Index>();
        const auto& keysShape = Input(0).sizes();
        CAFFE_ENFORCE_EQ(
            keysShape.size(), 1, "Only 1D keys tensor supported currently.");

        // 1. Shape and type consistency checks
        const auto& in0Shape = Input(1).sizes();
        CAFFE_ENFORCE_GE(in0Shape.size(), 1);

        vector<int64_t> outShape(keysShape.vec());
        outShape.insert(outShape.end(), in0Shape.begin() + 1, in0Shape.end());

        CAFFE_ENFORCE_GE(outShape.size(), 1);
        auto totalSize = in0Shape[0];
        auto meta = Input(1).dtype();
        for (int i = 2; i < InputSize(); ++i) {
          const auto& input = Input(i);
          CAFFE_ENFORCE(meta == input.dtype());
          CAFFE_ENFORCE_GE(input.dim(), 1);
          CAFFE_ENFORCE(std::equal(
              outShape.begin() + keysShape.size(),
              outShape.end(),
              input.sizes().begin() + 1));
          totalSize += input.size(0);
        }
        CAFFE_ENFORCE_EQ(keysTensor.numel(), totalSize);

        auto* outTensor = Output(0);
        outTensor->Resize(outShape);
        auto* outData = static_cast<char*>(outTensor->raw_mutable_data(meta));
        const auto blockSize = outTensor->size_from_dim(1);

        inputDatas_.resize(numPartitions);
        for (int i = 0; i < numPartitions; ++i) {
          inputDatas_[i] = static_cast<const char*>(Input(i + 1).raw_data());
        }
        inStartOffsets_.assign(numPartitions, 0);
        Index outStartOffset = 0;
        int currentShard = -1;

        // 2. copy from inputs into output based on shard for each input key
        const auto numEntries = keysTensor.numel();
        for (int64_t i = 0; i <= numEntries; ++i) {
          auto newShard =
              i < numEntries ? moduloPartition(keysData[i], numPartitions) : -1;
          if (newShard != currentShard) {
            if (currentShard != -1) {
              auto inStartOffset = inStartOffsets_[currentShard];
              auto numItems = i - outStartOffset;
              context_.CopyItemsSameDevice(
                  meta,
                  numItems * blockSize,
                  inputDatas_[currentShard] +
                      inStartOffset * blockSize * meta.itemsize(),
                  outData + outStartOffset * blockSize * meta.itemsize());
              inStartOffsets_[currentShard] += numItems;
            }
            currentShard = newShard;
            outStartOffset = i;
          }
        }

        return true;
        */
    }
}
