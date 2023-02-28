crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    CPUContext,
    TypeMeta,
    OperatorDef,
};

#[inline] pub fn modulo_partition<Index>(key: Index, num_partitions: i32) -> i32 {
    todo!();
    /*
        int shard = key % numPartitions;
      // equivalent to `if (shard < 0) shard += partitions;`
      shard += numPartitions & (shard >> (sizeof(int) * 8 - 1));
      return shard;
    */
}

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
pub struct GatherByKeyOp {
    //USE_DISPATCH_HELPER
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    storage: OperatorStorage,
    context: CPUContext,

    input_datas:      Vec<*mut u8>,
    in_start_offsets: Vec<i64>,
}

num_inputs!{GatherByKey, (2,INT_MAX)}

num_outputs!{GatherByKey, 1}

inputs!{GatherByKey, 
    0 => ("keys", "The first input is the full keys tensor (same as the first input of Partition)."),
    1 => ("sharded_values", "Subsequented inputs are sharded values tensors.")
}

outputs!{GatherByKey, 
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
pub struct PartitionOpBase {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    storage: OperatorStorage,
    context: CPUContext,

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

///------------------------------------------
pub struct PartitionOp {
    //USE_DISPATCH_HELPER
    base: PartitionOpBase,
}

impl PartitionOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : PartitionOpBase(std::forward<Args>(args)...)
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
            ApplyPartition<Index>(false /* skipFirstArgument */);
        return true;
        */
    }
}

/**
  | LengthsPartition splits the input
  | int tensor into multiple ones according
  | to the second tensor. The first dimension
  | is expected to be the tensor that describes
  | lengths of the elements.
  | 
  | Takes the second input and partitions
  | it to shards according to the remainder
  | of values modulo the number of partitions.
  | It requires the second tensor to be a
  | 1D-tensor of the integral type. The
  | first tensor should be 1D-tensor of
  | int32 that would represent the lengths
  | of the elements in the input. The number
  | of partitions is derived as (num_output
  | / num_input).
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
pub struct LengthsPartitionOp {
    //USE_DISPATCH_HELPER
    base:       PartitionOpBase,
    out_length: Vec<*mut i32>,
}

num_inputs_outputs!{LengthsPartition, 
    |input: i32, output: i32| {
        input >= 2 && output > 0 && output % input == 0
    }
}

inputs!{LengthsPartition, 
    0 => ("input", "Input tensor containing data to be partitioned. The number of input tensors might be greater than 1 but must have the same shape as the previous tensors.")
}

outputs!{LengthsPartition, 
    0 => ("partitions", "Output Partitions. The number of output tensors has to be a multiple of the number of input tensors.")
}

args!{LengthsPartition, 
    0 => ("pack_first_input", "(int, default 0) If set, the operator transforms the first tensor values as floor(X_ij / num_partitions)")
}

impl LengthsPartitionOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : PartitionOpBase(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(1));
        */
    }
    
    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE(
            OutputSize() % InputSize() == 0,
            "Output number must be a multiple of input number");
        int partitions = OutputSize() / InputSize();
        CAFFE_ENFORCE_GT(partitions, 0, "Invalid number of partitions");
        CAFFE_ENFORCE_EQ(
            Input(1).dim(),
            1,
            "Only 1-D tensors supported as a partitioning tensor for sharding");

        if (partitions == 1) {
          // Specialization when partitions == 1 which just becomes a copy.
          for (int i = 0; i < InputSize(); ++i) {
            auto& input = Input(i);
            auto& output = *Output(i);
            output.ResizeLike(input);
            context_.CopyItemsSameDevice(
                input.dtype(),
                input.numel(),
                input.raw_data(),
                output.raw_mutable_data(input.dtype()));
          }
          return true;
        }

        // Apply sharding to all parameters except lengths
        ApplyPartition<Index>(true /* skipFirstArgument */);

        // Compute lengths after sharding
        auto& main_input = Input(1);
        int64_t size = main_input.numel();
        const Index* data = main_input.template data<Index>();

        auto& length_input = Input(0);
        int64_t elements = length_input.numel();
        const int32_t* lengths_data = length_input.template data<int32_t>();
        out_length_.resize(partitions);
        for (int i = 0; i < partitions; ++i) {
          auto& output = *Output(i * InputSize());
          output.Resize(elements);
          out_length_[i] = output.template mutable_data<int32_t>();
        }

        int total_length = 0;
        for (int i = 0; i < elements; ++i) {
          total_length += lengths_data[i];
        }
        CAFFE_ENFORCE(
            total_length == size,
            "Total length is not matching to the number of elements");

        int index = 0;
        for (int i = 0; i < elements; ++i) {
          for (int j = 0; j < partitions; ++j) {
            out_length_[j][i] = 0;
          }
          for (int j = 0; j < lengths_data[i]; ++j, ++index) {
            int shard = moduloPartition(data[index], partitions);
            ++out_length_[shard][i];
          }
        }
        return true;
        */
    }
}

register_cpu_operator!{Partition,        PartitionOp}
register_cpu_operator!{LengthsPartition, LengthsPartitionOp}
register_cpu_operator!{GatherByKey,      GatherByKeyOp}

///------------------------------------------
pub struct GetGatherByKeyGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetGatherByKeyGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        auto pack_first_input =
            argsHelper.GetSingleArgument<int>("pack_first_input", 0);

        Argument packArg = MakeArgument<int>("pack_first_input", pack_first_input);
        if (g_output_[0].IsDense()) {
          std::vector<std::string> inputs;
          for (int i = 1; i < g_input_.size(); ++i) {
            inputs.push_back("_" + GI(i) + "_keys");
            inputs.push_back(GI(i));
          }
          return SingleGradientDef(
              "Partition",
              "",
              std::vector<std::string>{I(0), GO(0)},
              inputs,
              std::vector<Argument>{packArg});
        } else {
          std::vector<std::string> inputs;
          for (int i = 1; i < g_input_.size(); ++i) {
            inputs.push_back("_" + GI_I(i) + "_keys");
            inputs.push_back(GI_I(i));
            inputs.push_back(GI_V(i));
          }
          return SingleGradientDef(
              "Partition",
              "",
              std::vector<std::string>{I(0), GO_I(0), GO_V(0)},
              inputs,
              std::vector<Argument>{packArg});
        }
        */
    }
}

/**
  | This should actually have gradient, but for
  | now nothing uses it.
  |
  | Because gradient computation right now is not
  | input/output aware it can't be
  | GRADIENT_NOT_IMPLEMENTEDYET
  */
no_gradient!{Partition}
no_gradient!{LengthsPartition}
register_gradient!{GatherByKey, GetGatherByKeyGradient}
