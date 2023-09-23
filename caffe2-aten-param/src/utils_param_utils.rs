crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/utils/ParamUtils.h]

#[inline] pub fn expand_param_if_needed(
    list_param:   &[i32],
    param_name:   *const u8,
    expected_dim: i64) -> Vec<i64> {

    todo!();
        /*
            if (list_param.size() == 1) {
        return vector<i64>(expected_dim, list_param[0]);
      } else if ((i64)list_param.size() != expected_dim) {
        ostringstream ss;
        ss << "expected " << param_name << " to be a single integer value or a "
           << "list of " << expected_dim << " values to match the convolution "
           << "dimensions, but got " << param_name << "=" << list_param;
        AT_ERROR(ss.str());
      } else {
        return list_param.vec();
      }
        */
}
