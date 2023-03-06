crate::ix!();

/**
  | Parses a flat list of parameter tensors
  | into a list of CellParams
  |
  */
#[inline] pub fn gather_params(
    params:     &Vec<Tensor>,
    has_biases: bool,
    context:    *mut CPUContext) -> Vec<CellParams> 
{
    todo!();
    /*
        Tensor undefined;
      std::vector<CellParams> result;
      if (has_biases) {
        CAFFE_ENFORCE_EQ(
            params.size() % 4, 0, "got an incorrect number of LSTM parameters");
        for (size_t i = 0; i < params.size(); i += 4) {
          result.emplace_back(
              params[i], params[i + 1], params[i + 2], params[i + 3], context);
        }
      } else {
        CAFFE_ENFORCE_EQ(
            params.size() % 2, 0, "got an incorrect number of LSTM parameters");
        for (size_t i = 0; i < params.size(); i += 2) {
          result.emplace_back(
              params[i], params[i + 1], undefined, undefined, context);
        }
      }
      return result;
    */
}
