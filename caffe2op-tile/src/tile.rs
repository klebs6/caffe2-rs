crate::ix!();

/**
  | Constructs a tensor by tiling a given
  | tensor along a specified axis.
  | 
  | This operation creates a new tensor
  | by replicating the input tensor a number
  | of times specified by the `tiles` argument
  | along the `axis` dimension.
  | 
  | The output tensor's `axis` dimension
  | has $(X.dims(axis) * tiles)$ elements.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cc
  | 
  | Copy a Blob n times along a specified
  | axis.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TileOp<Context> {
    storage: OperatorStorage,
    context: Context,
    tiles:   i32,
    axis:    i32,
}

num_inputs!{Tile, (1,3)}

num_outputs!{Tile, 1}

inputs!{Tile, 
    0 => ("X",     "(*Tensor*): input tensor"),
    1 => ("tiles", "(*Tensor`<int>`*): [OPTIONAL] number of replicas (overrides `tiles` argument)"),
    2 => ("axis",  "(*Tensor`<int>`*): [OPTIONAL] axis to replicate along (overrides `axis` argument)")
}

outputs!{Tile, 
    0 => ("Y", "(*Tensor*): output tensor")
}

args!{Tile, 
    0 => ("tiles", "(*int*): number of replicas"),
    1 => ("axis",  "(*int*): axis to replicate along")
}

tensor_inference_function!{
    Tile, 
    /* ([](const OperatorDef& def,
                                const std::vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      out[0] = TensorShape(in[0]);
      ArgumentHelper helper(def);
      const std::int32_t tiles =
          helper.GetSingleArgument<std::int32_t>("tiles", 1);
      const std::int32_t axis =
          helper.GetSingleArgument<std::int32_t>("axis", 0);
      if (in.size() > 1) {
        // Tile or axis is specified as input; we can't determine
        // the size
        out[0].set_unknown_shape(true);
      } else {
        const auto canonical_axis =
            canonical_axis_index_(axis, out[0].dims().size());
        out[0].set_dims(
            canonical_axis, out[0].dims().Get(canonical_axis) * tiles);
      }
      return out;
    }) */
}

inherit_onnx_schema!{Tile}

register_cpu_operator!{Tile, TileOp<CPUContext>}
