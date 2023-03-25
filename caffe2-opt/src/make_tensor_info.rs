crate::ix!();

#[inline] pub fn make_tensor_info_with_quantized_flag(
    t:         &Vec<TensorBoundShape_DimType>,
    dims:      &Vec<i64>,
    dtype:     Option<TensorProto_DataType>,
    quantized: Option<bool>) -> ShapeInfo 
{
    let dtype: TensorProto_DataType = dtype.unwrap_or(TensorProto_DataType::FLOAT);
    let quantized: bool = quantized.unwrap_or(false);

    todo!();
    /*
        ShapeInfo info;
      info.setDimType(t);
      TensorShape& shape = info.shape;
      for (const auto d : dims) {
        shape.add_dims(d);
      }
      shape.set_data_type(dtype);
      if (quantized) {
        info.is_quantized = true;
        info.q_info.scale.clear();
        info.q_info.scale.push_back(1);
        info.q_info.offset.clear();
        info.q_info.offset.push_back(0);
        info.q_info.axis = 1;
      }
      return info;
    */
}

pub fn make_tensor_info(
    t:     &Vec<TensorBoundShape_DimType>,
    dims:  &Vec<i64>,
    dtype: Option<TensorProto_DataType>) -> ShapeInfo 
{
    let dtype = dtype.unwrap_or(TensorProto_DataType::FLOAT);

    todo!();
    /*
      ShapeInfo info;
      info.setDimType(t);
      TensorShape& shape = info.shape;
      for (const auto d : dims) {
        shape.add_dims(d);
      }
      shape.set_data_type(dtype);
      return info;
    */
}


