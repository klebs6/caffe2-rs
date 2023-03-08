crate::ix!();

/**
  | Max computation is done element-wise,
  | so that each element of the output slice
  | corresponds to the max value of the respective
  | elements in the input slices. Operation
  | doesn't change the shape of individual
  | blocks. This implementation imitates
  | torch nn.Max operator.
  | 
  | If the maximum value occurs more than
  | once, the operator will return the first
  | occurrence of value. When computing
  | the gradient using the backward propagation,
  | the gradient input corresponding to
  | the first occurrence of the maximum
  | value will be used.
  |
  */
pub struct MaxRangeReducerDef {
    
    /*
      template <typename T, class Context>
      using Reducer = MaxRangeReducer<T, Context>;

      template <typename T, class Context>
      using ReducerGradient = MaxRangeReducerGradient<T, Context>;

      static constexpr const char* name = "Max";
    */
}
