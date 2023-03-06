crate::ix!();

#[inline] pub fn set_column_buffer_shape(
    n:                         i32,
    kernel_size:               i32,
    output_image_size:         i32,
    output_image_dims:         &Vec<i32>,
    order:                     StorageOrder,
    column_slice_dims:         *mut Vec<i32>,
    column_dims:               *mut Vec<i32>,
    column_transposed_dims:    *mut Vec<i32>,
    column_axes:               *mut Vec<i32>)  
{
    todo!();
    /*
        column_slice_dims->resize(output_image_dims.size() + 1);
      if (order == StorageOrder::NCHW) {
        column_slice_dims->front() = kernel_size;
        std::copy(
            output_image_dims.cbegin(),
            output_image_dims.cend(),
            column_slice_dims->begin() + 1);
      } else {
        std::copy(
            output_image_dims.cbegin(),
            output_image_dims.cend(),
            column_slice_dims->begin());
        column_slice_dims->back() = kernel_size;
      }
      *column_dims = order == StorageOrder::NCHW
          ? std::vector<int>{N, kernel_size, output_image_size}
          : std::vector<int>{N, output_image_size, kernel_size};
      *column_transposed_dims = order == StorageOrder::NCHW
          ? std::vector<int>{output_image_size, kernel_size, N}
          : std::vector<int>{output_image_size, N, kernel_size};
      *column_axes = order == StorageOrder::NCHW ? std::vector<int>{2, 1, 0}
                                                 : std::vector<int>{1, 0, 2};
    */
}
