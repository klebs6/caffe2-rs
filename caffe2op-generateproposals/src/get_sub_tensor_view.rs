crate::ix!();

/**
  | Get a sub tensor view from 'tensor' using
  | data pointer from 'tensor'
  |
  */
#[inline] pub fn get_sub_tensor_view<T>(tensor: &TensorCPU, dim0_start_index: i32) -> ConstTensorView<T> {

    todo!();
    /*
        DCHECK_EQ(tensor.dtype().itemsize(), sizeof(T));

      if (tensor.numel() == 0) {
        return utils::ConstTensorView<T>(nullptr, {});
      }

      std::vector<int> start_dims(tensor.dim(), 0);
      start_dims.at(0) = dim0_start_index;
      auto st_idx = ComputeStartIndex(tensor, start_dims);
      auto ptr = tensor.data<T>() + st_idx;

      auto input_dims = tensor.sizes();
      std::vector<int> ret_dims(input_dims.begin() + 1, input_dims.end());

      utils::ConstTensorView<T> ret(ptr, ret_dims);
      return ret;
    */
}
