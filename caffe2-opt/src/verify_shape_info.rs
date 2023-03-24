crate::ix!();

#[inline] pub fn verify_shape_info(
    info:      &ShapeInfoMap,
    name:      &String,
    t:         &Vec<TensorBoundShape_DimType>,
    dims:      &Vec<i64>,
    dtype:     Option<TensorProto_DataType>,
    quantized: Option<bool>)  
{
    let dtype: TensorProto_DataType = dtype.unwrap_or(TensorProto_DataType::FLOAT);
    let quantized: bool = quantized.unwrap_or(false);

    todo!();
    /*
        LOG(INFO) << "Checking " << name;
      const auto it = info.find(name);
      ASSERT_TRUE(it != info.end());
      const auto& shape_info = it->second;
      EXPECT_EQ(shape_info.getDimType(), t);
      const auto& shape = shape_info.shape;
      ASSERT_EQ(shape.dims_size(), dims.size());
      for (int i = 0; i < dims.size(); ++i) {
        EXPECT_EQ(dims[i], shape.dims(i));
      }
      EXPECT_EQ(shape.data_type(), dtype);
      EXPECT_EQ(shape_info.is_quantized, quantized);
    */
}
