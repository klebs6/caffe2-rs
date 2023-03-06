crate::ix!();

pub struct ShapeParams {
    n:                        i32,
    c:                        i32,
    m:                        i32,
    input_image_size:         i32,
    output_image_size:        i32,
    kernel_size:              i32,
    x_dims:                   Vec<i32>,
    column_slice_dims:        Vec<i32>,
    column_dims:              Vec<i32>,
    column_transposed_dims:   Vec<i32>,
    column_axes:              Vec<i32>,
    y_dims:                   Vec<i32>,
    y_transposed_dims:        Vec<i32>,
    y_axes:                   Vec<i32>,
}

pub struct CUDAConvNetShapeParams {
    n:   i32,
    c:   i32,
    m:   i32,
    x_H: i32,
    x_W: i32,
    y_H: i32,
    y_W: i32,
}
