crate::ix!();

/// Copy a Blob n times along a specified axis.
///
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NumpyTileOp<Context> {
    storage: OperatorStorage,
    context: Context,
    buffer:  Tensor, // {Context::GetDeviceType()};
}

register_cpu_operator!{NumpyTile, NumpyTileOp<CPUContext>}

num_inputs!{NumpyTile, 2}

inputs!{NumpyTile, 
    0 => ("data",    "The input tensor."),
    1 => ("repeats", "1-D Tensor specifying how many times to repeat each axis.")
}

outputs!{NumpyTile, 
    0 => ("tiled_data", "Tensor that will contain input replicated along the given axis.")
}

inherit_onnx_schema!{NumpyTile, "Tile"}
