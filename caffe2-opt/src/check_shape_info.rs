crate::ix!();

#[inline] pub fn check_shape_info(
    shape_map:          &mut ShapeInfoMap,
    expected_shape_map: &mut ShapeInfoMap)  
{
    
    todo!();
    /*
        CHECK_EQ(shape_map.size(), expected_shape_map.size());
      for (auto& [name, shape] : shape_map) {
        auto it = expected_shape_map.find(name);
        CHECK(it != expected_shape_map.end()) << "Didn't find name " << name;
        auto& expected_shape = it->second;
        EXPECT_EQ(expected_shape.getDimType(), shape.getDimType());
        ASSERT_EQ(expected_shape.shape.dims_size(), shape.shape.dims_size());
        for (int i = 0; i < shape.shape.dims_size(); ++i) {
          EXPECT_EQ(expected_shape.shape.dims(i), shape.shape.dims(i));
        }
        EXPECT_EQ(expected_shape.shape.data_type(), shape.shape.data_type()) << "Shapes don't match for " << name;
        EXPECT_EQ(expected_shape.is_quantized, shape.is_quantized);
      }
    */
}
