crate::ix!();

/**
  | Collect `DATA` tensor into `RESERVOIR`
  | of size `num_to_collect`. `DATA` is
  | assumed to be a batch.
  | 
  | In case where 'objects' may be repeated
  | in data and you only want at most one instance
  | of each 'object' in the reservoir, `OBJECT_ID`
  | can be given for deduplication. If `OBJECT_ID`
  | is given, then you also need to supply
  | additional book-keeping tensors.
  | See input blob documentation for details.
  | 
  | This operator is thread-safe.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ReservoirSamplingOp<Context> {
    storage: OperatorStorage,
    context: Context,

    /// number of tensors to collect
    num_to_collect:  i32,
}

register_cpu_operator!{ReservoirSampling, ReservoirSamplingOp<CPUContext>}

num_inputs!{ReservoirSampling, (4,7)}

num_outputs!{ReservoirSampling, (2,4)}

num_inputs_outputs!{ReservoirSampling, 
    |input: i32, output: i32| {
        input / 3 == output / 2
    }
}

inputs!{ReservoirSampling, 
    0 => ("RESERVOIR",            "The reservoir; should be initialized to empty tensor"),
    1 => ("NUM_VISITED",          "Number of examples seen so far; should be initialized to 0"),
    2 => ("DATA",                 "Tensor to collect from. The first dimension is assumed to be batch size. If the object to be collected is represented by multiple tensors, use `PackRecords` to pack them into single tensor."),
    3 => ("MUTEX",                "Mutex to prevent data race"),
    4 => ("OBJECT_ID",            "(Optional, int64) If provided, used for deduplicating object in the reservoir"),
    5 => ("OBJECT_TO_POS_MAP_IN", "(Optional) Auxiliary bookkeeping map. This should be created from  `CreateMap` with keys of type int64 and values of type int32"),
    6 => ("POS_TO_OBJECT_IN",     "(Optional) Tensor of type int64 used for bookkeeping in deduplication")
}

outputs!{ReservoirSampling, 
    0 => ("RESERVOIR",           "Same as the input"),
    1 => ("NUM_VISITED",         "Same as the input"),
    2 => ("OBJECT_TO_POS_MAP",   "(Optional) Same as the input"),
    3 => ("POS_TO_OBJECT",       "(Optional) Same as the input")
}

args!{ReservoirSampling, 
    0 => ("num_to_collect", "The number of random samples to append for each positive samples")
}

enforce_inplace!{ReservoirSampling, vec![(0, 0), (1, 1), (5, 2), (6, 3)]}

should_not_do_gradient!{ReservoirSampling}

input_tags!{
    ReservoirSamplingOp {
        ReservoirIn,
        NumVisitedIn,
        Data,
        Mutex,
        ObjectId,
        ObjectToPosMapIn,
        PosToObjectIn
    }
}

output_tags!{
    ReservoirSamplingOp {
        Reservoir,
        NumVisited,
        ObjectToPosMap,
        PosToObject
    }
}
