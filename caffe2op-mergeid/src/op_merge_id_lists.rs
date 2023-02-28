crate::ix!();

use crate::{
    OperatorStorage,
};

/**
  | MergeIdLists: Merge multiple ID_LISTs
  | into a single ID_LIST.
  | 
  | An ID_LIST is a list of IDs (may be ints,
  | often longs) that represents a single
  | feature. As described in https://caffe2.ai/docs/sparse-operations.html,
  | a batch of ID_LIST examples is represented
  | as a pair of lengths and values where
  | the `lengths` (int32) segment the `values`
  | or ids (int32/int64) into examples.
  | 
  | Given multiple inputs of the form lengths_0,
  | values_0, lengths_1, values_1, ...
  | which correspond to lengths and values
  | of ID_LISTs of different features,
  | this operator produces a merged ID_LIST
  | that combines the ID_LIST features.
  | The final merged output is described
  | by a lengths and values vector.
  | 
  | WARNING: The merge makes no guarantee
  | about the relative order of ID_LISTs
  | within a batch. This can be an issue if
  | ID_LIST are order sensitive.
  |
  */
pub struct MergeIdListsOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
}

num_outputs!{MergeIdLists, 2}

num_inputs!{MergeIdLists, 
    |n: i32| {
        n > 0 && n % 2 == 0
    }
}

inputs!{MergeIdLists, 
    0 => ("lengths_0", "Lengths of the ID_LISTs batch for first feature"),
    1 => ("values_0", "Values of the ID_LISTs batch for first feature")
}

outputs!{MergeIdLists, 
    0 => ("merged_lengths", "Lengths of the merged ID_LISTs batch"),
    1 => ("merged_values", "Values of the merged ID_LISTs batch")
}

impl<Context> MergeIdListsOp<Context> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& first_lengths = Input(0);
        CAFFE_ENFORCE_EQ(first_lengths.dim(), 1, "LENGTHS should be 1-D");
        const auto batch_size = first_lengths.numel();

        auto* out_lengths = Output(0, first_lengths.sizes(), at::dtype<int32_t>());

        auto* out_lengths_data = out_lengths->template mutable_data<int32_t>();

        /**
         * Loop to figure out how much space to reserve for output
         * and perform checks.
         */
        auto M = 0;
        for (size_t i = 0; i < InputSize(); i += 2) {
          auto& lengths = Input(i);
          CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS should be 1-D");
          CAFFE_ENFORCE_EQ(lengths.numel(), batch_size, "LENGTHS should be equal");
          auto& values = Input(i + 1);
          CAFFE_ENFORCE_EQ(values.dim(), 1, "VALUES should be 1-D");
          M += values.numel();
        }

        auto* out_values = Output(1, {M}, at::dtype<T>());

        T* out_values_data = out_values->template mutable_data<T>();
        auto pos = 0;

        // TODO(badri): Use unordered_set if performance is an issue
        std::set<T> deduped;
        std::vector<int> offsets(InputSize(), 0);
        for (auto sample = 0; sample < batch_size; sample++) {
          for (size_t i = 0; i < InputSize(); i += 2) {
            auto& lengths = Input(i);
            const auto* lengths_data = lengths.template data<int32_t>();

            auto& values = Input(i + 1);
            const T* values_data = values.template data<T>();
            const auto length = lengths_data[sample];

            for (auto j = offsets[i]; j < offsets[i] + length; j++) {
              deduped.insert(values_data[j]);
            }
            offsets[i] += length;
          }
          for (auto val : deduped) {
            out_values_data[pos++] = val;
          }
          out_lengths_data[sample] = deduped.size();
          deduped.clear();
        }
        out_values->Resize(pos);
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(1));
        */
    }
}

register_cpu_operator!{MergeIdLists, MergeIdListsOp<CPUContext>}

no_gradient!{MergeIdLists}
