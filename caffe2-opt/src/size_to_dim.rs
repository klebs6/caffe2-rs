crate::ix!();

#[inline] pub fn set_dim_type_with_first(
    first_dim_type: TensorBoundShape_DimType, 
    n: u32) -> Vec<TensorBoundShape_DimType> 
{
    todo!();
    /*
        std::vector<TensorBoundShape_DimType> dimTypes(
          n, TensorBoundShape_DimType_CONSTANT);
      if (dimTypes.size() > 0) {
        dimTypes[0] = firstDimType;
      }
      return dimTypes;
    */
}

#[inline] pub fn size_from_dim(shape: &TensorShape, axis: i32) -> i64 {
    
    todo!();
    /*
        int64_t r = 1;
      for (int i = axis; i < shape.dims_size(); ++i) {
        r *= shape.dims(i);
      }
      return r;
    */
}


#[inline] pub fn size_to_dim(shape: &TensorShape, axis: i32) -> i64 {
    
    todo!();
    /*
        CAFFE_ENFORCE_LE(axis, shape.dims_size());
      int64_t r = 1;
      for (int i = 0; i < axis; ++i) {
        r *= shape.dims(i);
      }
      return r;
    */
}


