crate::ix!();

/**
  | TensorProtosDBInput is a simple input
  | operator that basically reads things
  | from a db where each key-value pair stores
  | an index as key, and a TensorProtos object
  | as value. These TensorProtos objects
  | should have the same size, and they will
  | be grouped into batches of the given
  | size. The DB Reader is provided as input to 
  | the operator and it returns as many output 
  | tensors as the size of the
  | 
  | TensorProtos object. Each output will
  | simply be a tensor containing a batch
  | of data with size specified by the 'batch_size'
  | argument containing data from the corresponding
  | index in the
  | 
  | TensorProtos objects in the DB.
  |
  */
pub struct TensorProtosDBInput<Context> {

    base:              PrefetchOperator<Context>,

    /**
      | Prefetch will always just happen on
      | the CPU side.
      |
      */
    prefetched_blobs:  Vec<Blob>,
    batch_size:        i32,
    shape_inferred:    bool, // default = false
    key:               String,
    value:             String,
}

register_cpu_operator!{TensorProtosDBInput, TensorProtosDBInput<CPUContext>}

num_inputs!{TensorProtosDBInput, 1}

num_outputs!{TensorProtosDBInput, (1,INT_MAX)}

inputs!{TensorProtosDBInput, 
    0 => ("data", "A pre-initialized DB reader. Typically, this is obtained by calling CreateDB operator with a db_name and a db_type. The resulting output blob is a DB Reader tensor")
}

outputs!{TensorProtosDBInput, 
    0 => ("output", "The output tensor in which the batches of data are returned. The number of output tensors is equal to the size of (number of TensorProto's in) the TensorProtos objects stored in the DB as values. Each output tensor will be of size specified by the 'batch_size' argument of the operator")
}

args!{TensorProtosDBInput, 
    0 => ("batch_size", "(int, default 0) the number of samples in a batch. The default value of 0 means that the operator will attempt to insert the entire data in a single output blob.")
}

no_gradient!{TensorProtosDBInput}

register_cuda_operator!{TensorProtosDBInput, TensorProtosDBInput<CUDAContext>}

