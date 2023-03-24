crate::ix!();

/**
  | Check precedence between two vector of
  | TensorBoundShape::DimType.
  |
  | If return 1: right take precedence over left
  | If return -1: left take precedence over right
  | If return 0: no precedence between left and right
  */
#[inline] pub fn take_precedence_over(
    left: &Vec<TensorBoundShape_DimType>, 
    right: &Vec<TensorBoundShape_DimType>) -> i32 
{
    todo!();
    /*
        const static std::vector<
          std::tuple<TensorBoundShape_DimType, TensorBoundShape_DimType, int>>
          precedence = {
              std::tuple<TensorBoundShape_DimType, TensorBoundShape_DimType, int>{
                  TensorBoundShape_DimType_FEATURE_MAX_DEFAULT,
                  TensorBoundShape_DimType_FEATURE_MAX,
                  1},
              std::tuple<TensorBoundShape_DimType, TensorBoundShape_DimType, int>{
                  TensorBoundShape_DimType_FEATURE_MAX,
                  TensorBoundShape_DimType_FEATURE_MAX_DEFAULT,
                  -1},
              std::tuple<TensorBoundShape_DimType, TensorBoundShape_DimType, int>{
                  TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT,
                  TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX,
                  1},
              std::tuple<TensorBoundShape_DimType, TensorBoundShape_DimType, int>{
                  TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX,
                  TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT,
                  -1}};

      // If left is empty and right is not, right take precedence
      if (left.size() == 0 || right.size() == 0) {
        return right.size() > left.size();
      }
      for (int i = 0; i < right.size(); i++) {
        // If right.size > left.size and left[0:i] == right[0:i],
        // right take precedence
        if (i >= left.size()) {
          return 1;
        }
        auto l = left[i];
        auto r = right[i];
        if (l == TensorBoundShape_DimType_UNKNOWN &&
            r != TensorBoundShape_DimType_UNKNOWN) {
          return 1;
        }
        if (r == TensorBoundShape_DimType_UNKNOWN &&
            l != TensorBoundShape_DimType_UNKNOWN) {
          return -1;
        }
        for (auto& t : precedence) {
          if (l == std::get<0>(t) && r == std::get<1>(t)) {
            return std::get<2>(t);
          }
        }
        if (l != r) {
          return 0;
        }
      }
      return 0;
    */
}



