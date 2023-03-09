crate::ix!();

/**
  | Deduplicates input indices vector
  | and optionally produces reverse remapping.
  | 
  | There's no guarantees on the ordering
  | of the output indices.
  | 
  | Current implementation produces a
  | sorted list but it's not guaranteed
  | in general.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UniqueOp<Context> {
    storage:              OperatorStorage,
    context:              Context,
    order:                Vec<i32>,
    thrust_unique_buffer: Tensor,
    cuda_order_buffer:    Tensor, // {Context::GetDeviceType()};
    second_order_buffer:  Tensor, // {Context::GetDeviceType()};
}

num_inputs!{Unique, 1}

num_outputs!{Unique, (1,2)}

inputs!{Unique, 
    0 => ("indices", "1D tensor of int32 or int64 indices.")
}

outputs!{Unique, 
    0 => ("unique_indices", "1D tensor of deduped entries."),
    1 => ("remapping",      "(optional) mapping from `indices` to `unique_indices`. This has the same shape as `indices`. Its elements are the indices into `unique_indices` such that `Gather(['unique_indices', 'remapping'])` yields `indices`.")
}

tensor_inference_function!{
    Unique, 
    /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      CAFFE_ENFORCE_EQ(in[0].dims_size(), 1);
      if (in[0].dims(0) <= 1) {
        // This special case is useful in some situation, e.g., when feeding
        // tensor inference with empty tensor (where the first dim is the batch
        // size)
        out[0].add_dims(in[0].dims(0));
      } else {
        out[0].set_unknown_shape(true);
      }
      if (def.output_size() > 1) {
        // Remapping has the same shape as the input tensor
        out.push_back(in[0]);
        out.back().set_data_type(TensorProto::INT32);
      }
      return out;
    } */}


should_not_do_gradient!{Unique}

output_tags!{
    UniqueOp {
        Unique,
        Remapping
    }
}

impl<Context> UniqueOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }
}
