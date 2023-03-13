crate::ix!();

/**
  | Check that indices fall within dimension
  | array size with CAFFE_ENFORCE.
  |
  */
#[inline] pub fn check_indexarray_range<IndexType>(
    indices:           *const IndexType,
    n:                 i64,
    indexing_axis_dim: IndexType,
    wrap_indices:      bool) 
{
    todo!();
    /*
        //
      for (auto i = 0; i < n; ++i) {
        auto idx = indices[i];
        if (wrap_indices && idx < 0) {
          idx = idx + indexing_axis_dim;
        }
        CAFFE_ENFORCE(
            0 <= idx && idx < indexing_axis_dim,
            "INDICES element is out of DATA bounds, id=",
            idx,
            " axis_dim=",
            indexing_axis_dim);
      }
    */
}
