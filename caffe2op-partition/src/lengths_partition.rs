crate::ix!();

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
#[USE_DISPATCH_HELPER]
pub struct LengthsPartitionOp {
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
