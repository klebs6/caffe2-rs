crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/RowwisePrune.cpp]

pub fn rowwise_prune_helper<input_t>(
    weights:                  &Tensor,
    mask:                     &Tensor,
    compressed_indices_dtype: ScalarType) -> (Tensor,Tensor) {

    todo!();
        /*
            int num_non_masked_rows = 0;
      auto mask_contig = mask.contiguous();
      auto mask_data = mask_contig.data_ptr<bool>();
      for (int i = 0; i < mask.numel(); ++i) {
        num_non_masked_rows += (((mask_data[i] == true)) ? 1 : 0);
      }
      int num_cols = weights.size(1);
      auto pruned_2d_tensor = empty({num_non_masked_rows, num_cols},
          weights.options());
      auto compressed_indices_mapping = empty({mask.numel()},
          compressed_indices_dtype);
      AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Half,
                                 ScalarType::BFloat16,
                                 weights.scalar_type(),
                                "rowwise_prune_helper", [&]() {
        auto* pruned_2d_tensor_data = pruned_2d_tensor.data_ptr<Scalar>();
        auto compressed_indices_mapping_data =
            compressed_indices_mapping.data_ptr<input_t>();
        auto weights_data = weights.data_ptr<Scalar>();
        int last_row_kept = 0;
        for (int i = 0; i < mask.numel(); i++) {
          if (mask_data[i]) {
            memcpy(pruned_2d_tensor_data + last_row_kept * num_cols,
                  weights_data + i * num_cols,
                  num_cols * sizeof (Scalar));
            compressed_indices_mapping_data[i] = last_row_kept;
            last_row_kept++;
          } else {
            compressed_indices_mapping_data[i] = -1;
          }
        }
      });
      return tuple<Tensor, Tensor>(pruned_2d_tensor,
          compressed_indices_mapping);
        */
}

/**
  | This operator introduces sparsity to the
  | 'weights' matrix with the help of the
  | importance indicator 'mask'.
  |
  | A row is considered important and not pruned if
  | the mask value for that particular row is
  | 1(True) and not important otherwise.
  |
  | This operator doesn't zero out the pruned rows
  | in-place. Instead, it returns a tuple that
  | contains a pruned weights tensor as well as
  | a map that can be used to look up the original
  | row in the pruned weights tensor.
  |
  | We refer this map as 'compressed indices map'
  | going forward.
  | The 'compressed indices map' is an 1D tensor
  | that contains one entry per original row in
  | 'weights'.
  |
  | The array index is the index for the original
  | non-pruned weight tensor and the value would be
  | the re-mapped index in the pruned weights
  | tensor.
  |
  | If the value for a index is -1, it means the
  | corresponding row has been pruned from the
  | original weight tensor.
  | Arguments:
  |
  | 'weights' - two dimensional matrix that needs
  | to be prune.
  |
  | 'mask' - 1D boolean tensor that represents
  |    whether a row is important or not. A mask
  |    value of 1 means the row should be kept and
  |    0 means the row should be pruned.
  |
  | Returns:
  |
  | A tuple containing two tensors,
  |
  | 1. A pruned weight tensor that contains only
  |    the weights that are preserved post pruning.
  |
  | 2. An 1D tensor that contains the mapping
  |    between original weight row and the
  |    corresponding row in the pruned weights
  |    tensor.
  |
  */
pub fn rowwise_prune(
        weights:                  &Tensor,
        mask:                     &Tensor,
        compressed_indices_dtype: ScalarType) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(weights.ndimension() == 2,
          "'weights' should have 2 dimensions.");
      TORCH_CHECK(
        mask.numel() == weights.size(0),
        "Number of elements in 'mask' should be equivalent to the "
        "number of rows in 'weights'."
      )
      TORCH_CHECK(
          compressed_indices_dtype == ScalarType::Int ||
          compressed_indices_dtype == ScalarType::Long,
          "compressed_indices_dtype should be either int(int32) or long(int64).");

      if (compressed_indices_dtype == ScalarType::Int) {
        return _rowwise_prune_helper<i32>(weights, mask,
                                              compressed_indices_dtype);
      }
      return _rowwise_prune_helper<i64>(weights, mask,
                                            compressed_indices_dtype);
        */
}
