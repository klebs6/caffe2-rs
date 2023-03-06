crate::ix!();

#[inline] pub fn set_ybuffer_shape(
    n:                   i32,
    m:                   i32,
    output_image_size:   i32,
    order:               StorageOrder,
    y_dims:              *mut Vec<i32>,
    y_transposed_dims:   *mut Vec<i32>,
    y_axes:              *mut Vec<i32>)  
{
    todo!();
    /*
        *Y_dims = order == StorageOrder::NCHW
          ? std::vector<int>{N, M, output_image_size}
          : std::vector<int>{N, output_image_size, M};
      *Y_transposed_dims = order == StorageOrder::NCHW
          ? std::vector<int>{output_image_size, M, N}
          : std::vector<int>{output_image_size, N, M};
      *Y_axes = order == StorageOrder::NCHW ? std::vector<int>{2, 1, 0}
                                            : std::vector<int>{1, 0, 2};
    */
}
