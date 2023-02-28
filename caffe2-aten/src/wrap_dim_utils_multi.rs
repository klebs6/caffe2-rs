crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/WrapDimUtilsMulti.h]

/**
  | This is in an extra file to work around strange
  | interaction of bitset on Windows with operator
  | overloading
  |
  */
pub const DIM_BITSET_SIZE: usize = 64;

#[inline] pub fn dim_list_to_bitset(
    dims:  &[i32],
    ndims: i64) -> BitSet<DimBitsetSize> {
    
    todo!();
        /*
            TORCH_CHECK(ndims <= (i64) dim_bitset_size, "only tensors with up to ", dim_bitset_size, " dims are supported");
      bitset<dim_bitset_size> seen;
      for (usize i = 0; i < dims.size(); i++) {
        usize dim = maybe_wrap_dim(dims[i], ndims);
        TORCH_CHECK(!seen[dim], "dim ", dim, " appears multiple times in the list of dims");
        seen[dim] = true;
      }
      return seen;
        */
}
