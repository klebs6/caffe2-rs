crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cudnn/RNNUtils.h]


/// Declares utilities used by RNN.cpp and also
/// needed by external consumers
///
pub fn copy_weights_to_flat_buf_views(
        weight_arr:                   TensorList,
        weight_stride0:               i64,
        input_size:                   i64,
        mode:                         i64,
        hidden_size:                  i64,
        proj_size:                    i64,
        num_layers:                   i64,
        batch_first:                  bool,
        bidirectional:                bool,
        flat_buf_datatype:            CudnnDataType,
        flat_buf_options:             &TensorOptions,
        set_orig_weights_to_flat_buf: bool,
        allow_type_change:            bool,
        include_bias:                 bool) -> (Tensor,Vec<Tensor>) {

    let allow_type_change: bool = allow_type_change.unwrap_or(false);
    let include_bias: bool = include_bias.unwrap_or(true);

    todo!();
        /*
        
        */
}
