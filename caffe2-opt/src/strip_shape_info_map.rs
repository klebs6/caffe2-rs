crate::ix!();

/// Convert ShapeInfo map to TensorShape map
#[inline] pub fn strip_shape_info_map(info_map: &ShapeInfoMap) -> HashMap<String,TensorShape> {
    
    todo!();
    /*
        std::unordered_map<std::string, TensorShape> shape_map;
      for (const auto& kv : info_map) {
        shape_map.emplace(kv.first, kv.second.shape);
      }
      return shape_map;
    */
}
