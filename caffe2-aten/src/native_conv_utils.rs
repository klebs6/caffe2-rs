crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ConvUtils.h]

pub const INPUT_BATCH_SIZE_DIM:       i32 = 0; // also grad_input
pub const INPUT_CHANNELS_DIM:         i32 = 1;
pub const OUTPUT_BATCH_SIZE_DIM:      i32 = 0; // also grad_output
pub const OUTPUT_CHANNELS_DIM:        i32 = 1;
pub const WEIGHT_OUTPUT_CHANNELS_DIM: i32 = 0;
pub const WEIGHT_INPUT_CHANNELS_DIM:  i32 = 1;

/**
  | Often written as 2 + max_dim (extra dims
  | for batch size and channels)
  |
  */
pub const MAX_DIM: i32 = 3;

/**
  | NB: conv_output_size and conv_input_size are
  | not bijections, as conv_output_size loses
  | information; this is why conv_input_size takes
  | an extra output_padding argument to resolve the
  | ambiguity.
  */
#[inline] pub fn conv_output_size(
    input_size:  &[i32],
    weight_size: &[i32],
    padding:     &[i32],
    stride:      &[i32],
    dilation:    &[i32]) -> Vec<i64> {

    let dilation: &[i32] = dilation.unwrap_or(IntArrayRef);

    todo!();
        /*
            // ASSERT(input_size.size() > 2)
      // ASSERT(input_size.size() == weight_size.size())
      bool has_dilation = dilation.size() > 0;
      auto dim = input_size.size();
      std::vector<i64> output_size(dim);
      output_size[0] = input_size[input_batch_size_dim];
      output_size[1] = weight_size[weight_output_channels_dim];
      for (usize d = 2; d < dim; ++d) {
        auto dilation_ = has_dilation ? dilation[d - 2] : 1;
        auto kernel = dilation_ * (weight_size[d] - 1) + 1;
        output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
      }
      return output_size;
        */
}

#[inline] pub fn conv_input_size(
    output_size:    &[i32],
    weight_size:    &[i32],
    padding:        &[i32],
    output_padding: &[i32],
    stride:         &[i32],
    dilation:       &[i32],
    groups:         i64) -> Vec<i64> {

    todo!();
        /*
            // ASSERT(output_size.size() > 2)
      // ASSERT(output_size.size() == weight_size.size())
      auto dim = output_size.size();
      std::vector<i64> input_size(dim);
      input_size[0] = output_size[output_batch_size_dim];
      input_size[1] = weight_size[weight_input_channels_dim] * groups;
      for (usize d = 2; d < dim; ++d) {
        int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
        input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                         kernel + output_padding[d - 2];
      }
      return input_size;
        */
}

#[inline] pub fn conv_weight_size(
    input_size:     &[i32],
    output_size:    &[i32],
    padding:        &[i32],
    output_padding: &[i32],
    stride:         &[i32],
    dilation:       &[i32],
    groups:         i64) -> Vec<i64> {

    todo!();
        /*
            auto dim = input_size.size();
      std::vector<i64> weight_size(dim);
      weight_size[0] = output_size[1];
      weight_size[1] = input_size[1] / groups;
      for (usize d = 2; d < dim; ++d) {
        int kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
                   + 2 * padding[d - 2] - output_padding[d - 2];
        weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
      }
      return weight_size;
        */
}

#[inline] pub fn reshape_bias(
        dim:  i64,
        bias: &Tensor) -> Tensor {
    
    todo!();
        /*
            std::vector<i64> shape(dim, 1);
      shape[1] = -1;
      return bias.reshape(shape);
        */
}

#[inline] pub fn cudnn_conv_use_channels_last(
        input:  &Tensor,
        weight: &Tensor) -> bool {
    
    todo!();
        /*
            // disable NHWC for float64 input.
      if (!detail::getCUDAHooks().compiledWithCuDNN() ||
          input.scalar_type() == at::kDouble ||
          weight.scalar_type() == at::kDouble) {
        return false;
      }
      long cudnn_version = detail::getCUDAHooks().versionCuDNN();
      auto input_memory_format = input.suggest_memory_format();
      auto weight_memory_format = weight.suggest_memory_format();

      bool can_use_cudnn_channels_last_2d = (cudnn_version >= 7603) && (
        (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
        (weight_memory_format == at::MemoryFormat::ChannelsLast)
      );

      bool can_use_cudnn_channels_last_3d = (cudnn_version >= 8005) && (
        (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
        (weight_memory_format == at::MemoryFormat::ChannelsLast3d)
      );

      return can_use_cudnn_channels_last_2d || can_use_cudnn_channels_last_3d;
        */
}
