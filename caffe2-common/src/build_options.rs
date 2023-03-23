crate::ix!();

/**
  | Returns which setting Caffe2 was configured
  | and built with (exported from CMake)
  |
  */
pub fn get_build_options<'a>() -> &'a HashMap<String,String> {
    todo!();
    /*
      static const std::map<string, string> kMap = CAFFE2_BUILD_STRINGS;
      return kMap;
    */
}
