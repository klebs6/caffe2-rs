crate::ix!();

/**
  | Splits the input int tensor into multiple
  | ones according to the first tensor.
  | 
  | Takes the first input and partitions
  | it to shards according to the remainder
  | of values modulo the number of partitions.
  | It requires that the first tensor is
  | of integral type. The number of partitions
  | is derived as (num_output / num_input).
  | 
  | If additional inputs are present they
  | must have the same shape as the first
  | input, optionally with extra trailing
  | dimensions. They will be partitioned
  | accordingly to the first input.
  | 
  | Optional arg 'pack_first_input' transforms
  | the first tensor values as
  | 
  | X_ij / num_partitions.
  | 
  | Outputs are ordered as
  | 
  | X_0_part_0, X_1_part_0, ..., X_N-1_part_0,
  | X_0_part_1, ..., X_N-1_part_K-1
  |
  */
#[USE_OPERATOR_FUNCTIONS("CPUContext")]
pub struct PartitionOpBase {
    storage:              OperatorStorage,
    context:              CPUContext,
    pack_first_input:     bool,

    /// use member fields to reuse memory
    counts:               Vec<i64>,
    block_sizes:          Vec<i64>,
    metas:                Vec<TypeMeta>,
    raw_datas:            Vec<*const c_void>,
    out_datas:            Vec<*mut c_void>,
}

num_inputs_outputs!{Partition, 
    |input: i32, output: i32| {
        input > 0 && output > 0 && output % input == 0
    }
}

inputs!{Partition, 
    0 => ("input", "Input tensor containing data to be partitioned. The number of input tensors might be greater than 1 but must have the same shape as the previous tensors.")
}

outputs!{Partition, 
    0 => ("partitions", "Output Partitions. The number of output tensors has to be a multiple of the number of input tensors.")
}

args!{Partition, 
    0 => ("pack_first_input", "(int, default 0) If set, the operator transforms the first tensor values as floor(X_ij / num_partitions)")
}

impl PartitionOpBase {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "pack_first_input", pack_first_input_, 0)
        */
    }
    
    #[inline] pub fn apply_partition<Index>(&mut self, skip_first_argument: bool)  {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(
            OutputSize() % InputSize(),
            0,
            "Output number must be a multiple of input number");
        int partitions = OutputSize() / InputSize();
        int inputSize = InputSize();
        int mainInputIndex = skipFirstArgument;
        CAFFE_ENFORCE_GT(partitions, 0, "Invalid number of partitions");

        auto& main_input = Input(mainInputIndex);
        int64_t size = main_input.numel();
        const Index* data = main_input.template data<Index>();
        counts_.assign(partitions, 0);
        for (int64_t p = 0; p < size; p++) {
          int shard = moduloPartition(data[p], partitions);
          ++counts_[shard];
        }

        raw_datas_.resize(inputSize);
        block_sizes_.resize(inputSize);
        metas_.resize(inputSize);
        out_datas_.resize(OutputSize());
        for (int i = mainInputIndex; i < inputSize; ++i) {
          auto& input = Input(i);
          if (i > mainInputIndex) {
            CAFFE_ENFORCE_GE(
                input.dim(),
                main_input.dim(),
                "Prefix of extra input's shape must match main input's shape, ",
                "input: ",
                i);
            for (int j = 0; j < main_input.dim(); ++j) {
              CAFFE_ENFORCE_GE(
                  input.size(j),
                  main_input.size(j),
                  "Prefix of extra input's shape must match main input's shape, ",
                  "input: ",
                  i,
                  ", dim ",
                  j);
            }
          }
          raw_datas_[i] = input.raw_data();
          block_sizes_[i] = input.size_from_dim(main_input.dim());
          metas_[i] = input.dtype();
          // shape = partition_size + suffix of input dims
          vector<int64_t> shape(
              input.sizes().begin() + main_input.dim() - 1, input.sizes().end());
          for (int j = 0; j < partitions; ++j) {
            int out_idx = i + j * inputSize;
            auto output = Output(out_idx);
            shape[0] = counts_[j];
            output->Resize(shape);
            out_datas_[out_idx] = output->raw_mutable_data(input.dtype());
          }
        }

        counts_.assign(partitions, 0);
        for (int64_t p = 0; p < size; p++) {
          int shard = moduloPartition(data[p], partitions);
          int64_t idx = counts_[shard]++;

          // special case first input
          static_cast<Index*>(out_datas_[shard * inputSize + mainInputIndex])[idx] =
              pack_first_input_ ? ((data[p] - shard) / partitions) : data[p];

          int baseIndex = shard * inputSize;
          for (int i = mainInputIndex + 1; i < inputSize; ++i) {
            auto bs = block_sizes_[i];
            auto meta = metas_[i];
            // special case for small bs?
            context_.CopyItemsSameDevice(
                meta,
                bs,
                static_cast<const char*>(raw_datas_[i]) + p * bs * meta.itemsize(),
                static_cast<char*>(out_datas_[baseIndex + i]) +
                    idx * bs * meta.itemsize());
          }
        }
        */
    }
}
