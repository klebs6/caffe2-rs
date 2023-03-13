crate::ix!();

/**
  | New shape is concatenation:
  |
  |  [data dims before axis] + [indices dims]
  |  + [data dims after axis]
  */
#[inline] pub fn calc_output_shape_vector<IndexType, DataDimsVec, IndexDimsVec>(
    data_dims:    &DataDimsVec,
    indices_dims: &IndexDimsVec,
    axis:         i32,
    match_outer:  bool) -> Vec<IndexType> 
{
    todo!();
    /*
        vector<IndexType> shape;
      // If the dimension we are indexing is empty, just use data_dims as shape.
      // This replicates behavior in (https://github.com/pytorch/pytorch/pull/13781)
      // needed to allow workflows with empty batch to succeed.
      if (data_dims[axis] == 0) {
        shape.insert(shape.end(), data_dims.begin(), data_dims.end());
      } else {
        shape.insert(shape.end(), data_dims.begin(), data_dims.begin() + axis);
        if (match_outer) {
          shape.insert(
              shape.end(), indices_dims.begin() + axis, indices_dims.end());
        } else {
          shape.insert(shape.end(), indices_dims.begin(), indices_dims.end());
        }
        shape.insert(shape.end(), data_dims.begin() + axis + 1, data_dims.end());
      }
      return shape;
    */
}
