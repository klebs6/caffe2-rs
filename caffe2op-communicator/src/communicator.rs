/*!
  | Communication operators do not have
  | default engines.
  |
  */

crate::ix!();

/**
  | Creates a common world for communication
  | operators.
  |
  */
register_cpu_operator!{CreateCommonWorld,   NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{CreateCommonWorld, NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{CreateCommonWorld}

num_inputs!{CreateCommonWorld, (0,1)}

num_outputs!{CreateCommonWorld, 1}

inputs!{CreateCommonWorld, 
    0 => ("kv_handler", "Key/value handler for rendezvous (optional).")
}

outputs!{CreateCommonWorld, 
    0 => ("comm_world", "A common world for collective operations.")
}

args!{CreateCommonWorld, 
    0 => ("size", "(int) size of the common world."),
    1 => ("rank", "(int) rank of this node in the common world.")
}

/**
  | Clones existing common world.
  |
  */
register_cpu_operator!{CloneCommonWorld,    NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{CloneCommonWorld,  NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{CloneCommonWorld}

num_inputs!{CloneCommonWorld, 1}

num_outputs!{CloneCommonWorld, 1}

inputs!{CloneCommonWorld, 
    0 => ("existing_comm_world", "Existing common world to clone.")
}

outputs!{CloneCommonWorld, 
    0 => ("comm_world", "A common world for collective operations.")
}

/**
  | Closes all connections managed by a
  | common world.
  |
  */
register_cpu_operator!{DestroyCommonWorld,  NoDefaultEngineOp<CPUContext>}

should_not_do_gradient!{DestroyCommonWorld}

num_inputs!{DestroyCommonWorld, 1}

num_outputs!{DestroyCommonWorld, 1}

inputs!{DestroyCommonWorld, 
    0 => ("common_world", "The common world to be destroyed.")
}

enforce_inplace!{DestroyCommonWorld, vec![(0, 0)]}

/**
  | Does a broadcast operation from the
  | root node to every other node. The tensor
  | on each node should have been pre-created
  | with the same shape and data type.
  |
  */
register_cpu_operator!{Broadcast,           NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{Broadcast,         NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{Broadcast}

num_inputs_outputs!{
    Broadcast, 
    |input: i32, out: i32| {
        input >= 2 && out == (input - 1)
    }
}

inputs!{Broadcast, 
    0 => ("comm_world", "The common world."),
    1 => ("X", "A tensor to be broadcasted.")
}

outputs!{Broadcast, 
    0 => ("X", "In-place as input 1.")
}

args!{Broadcast, 
    0 => ("root", "(int, default 0) the root to run broadcast from.")
}

enforce_inplace!{Broadcast, 
    |input: i32, out: i32| {
        (input - 1) == out
    }
}

identical_type_and_shape_of_input!{Broadcast, 0}

inputs_can_cross_devices!{Broadcast}

/**
  | Does a reduce operation from every node
  | to the root node.
  | 
  | Currently only Sum is supported.
  |
  */
register_cpu_operator!{Reduce,              NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{Reduce,            NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{Reduce}

num_inputs!{Reduce, 2}

num_outputs!{Reduce, 1}

inputs!{Reduce, 
    0 => ("comm_world", "The common world."),
    1 => ("X", "A tensor to be reduced.")
}

outputs!{Reduce, 
    0 => ("Y", "The reduced result on root, not set for other nodes.")
}

args!{Reduce, 
    0 => ("root", "(int, default 0) the root to run reduce into.")
}

identical_type_and_shape_of_input!{Reduce, 0}

inputs_can_cross_devices!{Reduce}

/**
  | Does an allgather operation among the
  | nodes.
  |
  */
register_cpu_operator!{Allgather,           NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{Allgather,         NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{Allgather}

num_inputs!{Allgather, (2,INT_MAX)}

num_outputs!{Allgather, 1}

inputs!{Allgather, 
    0 => ("comm_world", "The common world."),
    1 => ("X", "A tensor to be allgathered.")
}

outputs!{Allgather, 
    0 => ("Y", "The allgathered tensor, same on all nodes.")
}

inputs_can_cross_devices!{Allgather}

/**
  | Does an allreduce operation among the
  | nodes. Currently only Sum is supported.
  |
  */
register_cpu_operator!{Allreduce,           NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{Allreduce,         NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{Allreduce}

num_inputs_outputs!{Allreduce, 
    |input: i32, out: i32| {
      input >= 2 && out == (input - 1)
    }
}

inputs!{Allreduce, 
    0 => ("comm_world", "The common world."),
    1 => ("X", "A tensor to be allreduced.")
}

outputs!{Allreduce, 
    0 => ("Y", "The allreduced tensor, same on all nodes.")
}

enforce_inplace!{Allreduce, 
    |input: i32, output: i32| {
        (input - 1) == output
    }
}

identical_type_and_shape_of_input!{Allreduce, 0}

inputs_can_cross_devices!{Allreduce}

/**
  | Does reduce-scatter operation among
  | the nodes.
  | 
  | Currently only Sum is supported.
  |
  */
register_cpu_operator!{ReduceScatter, NoDefaultEngineOp<CPUContext>}

should_not_do_gradient!{ReduceScatter}

num_inputs_outputs!{ReduceScatter, 
    |input: i32, output: i32| {
        input >= 2 && output == (input - 1)
    }
}

inputs!{ReduceScatter, 
    0 => ("comm_world", "The common world."),
    1 => ("X", "A tensor to be reduce-scattered.")
}

outputs!{ReduceScatter, 
    0 => ("Y", "The reduced tensor, scattered on all nodes.")
}

enforce_inplace!{ReduceScatter, 
    |input: i32, output: i32| {
        (input - 1) == output
    }
}

identical_type_and_shape_of_input!{ReduceScatter, 0}

inputs_can_cross_devices!{ReduceScatter}

/**
  | Does a barrier operation among the nodes.
  |
  */
register_cpu_operator!{Barrier,             NoDefaultEngineOp<CPUContext>}

should_not_do_gradient!{Barrier}

num_inputs!{Barrier, 1}

inputs!{Barrier, 
    0 => ("comm_world", "The common world.")
}

/**
  | Sends the tensor to another node.
  |
  */
register_cpu_operator!{SendTensor,          NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{SendTensor,        NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{SendTensor}

num_inputs!{SendTensor, (2,4)}

num_outputs!{SendTensor, 0}

inputs!{SendTensor, 
    0 => ("comm_world", "The common world."),
    1 => ("X", "A tensor to be allgathered."),
    2 => ("dst", "An int CPUtensor of size 1 specifying the rank. If given, this overrides the 'to' argument of the op."),
    3 => ("tag", "An int CPUtensor of size 1 specifying the tag to send the tensor with. This overrides the 'tag' argument of the op.")
}

args!{SendTensor, 
    0 => ("dst", "The rank to send the tensor to."),
    1 => ("tag", "(int) a tag to send the tensor with."),
    2 => ("raw_buffer", "(bool) if set, only send the content and assume that the receiver has already known the tensor's shape and information.")
}

/**
  | Receives the tensor from another node.
  |
  */
register_cpu_operator!{ReceiveTensor,       NoDefaultEngineOp<CPUContext>}

register_cuda_operator!{ReceiveTensor,     NoDefaultEngineOp<CUDAContext>}

should_not_do_gradient!{ReceiveTensor}

num_inputs!{ReceiveTensor, (2,4)}

num_outputs!{ReceiveTensor, 3}

inputs!{ReceiveTensor, 
    0 => ("comm_world", "The common world."),
    1 => ("Y", "In-place output. If raw_buffer is specified, Y should have pre-allocated data and type.."),
    2 => ("src", "An int CPUtensor of size 1 specifying the rank. If given, this overrides the 'from' argument of the op."),
    3 => ("tag", "An int CPUtensor of size 1 specifying the tag to send the tensor with. This overrides the 'tag' argument of the op.")
}

outputs!{ReceiveTensor, 
    0 => ("Y", "The received tensor."),
    1 => ("src", "The sender that sent the message as a CPUTensor of size 1 and of type int."),
    2 => ("tag", "The tag that the message is sent with as a CPUTensor of size 1 and of type int.")
}

args!{ReceiveTensor, 
    0 => ("src", "(int) he rank to receive the tensor from."),
    1 => ("tag", "(int) a tag to receive the tensor with."),
    2 => ("raw_buffer", "(bool) if set, only send the content and assume that the receiver has already known the tensor's shape and information.")
}

enforce_inplace!{ReceiveTensor, vec![(1, 0)]}

allow_inplace!{ReceiveTensor, vec![(2, 1), (3, 2)]}
