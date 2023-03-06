crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeIdListsOp<Context> {
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

register_cpu_operator!{MergeIdLists, MergeIdListsOp<CPUContext>}

no_gradient!{MergeIdLists}
