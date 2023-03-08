crate::ix!();

/**
  | Calculate the quantile for the value
  | in the given list of tensors.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct QuantileOp<Context> {
    storage:  OperatorStorage,
    context:  Context,
    quantile: f32,
    abs:      bool,
    tol:      f32,
}

impl<Context> QuantileOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            quantile_(this->template GetSingleArgument<float>("quantile", -1.0)),
            abs_(this->template GetSingleArgument<bool>("abs", true)),
            tol_(this->template GetSingleArgument<float>("tol", 1e-3)) 

        CAFFE_ENFORCE_GE(
            quantile_,
            0,
            "input quantile should be ",
            "no less than 0, got ",
            quantile_);
        CAFFE_ENFORCE_GE(
            1.0f,
            quantile_,
            "input quantile should be ",
            "no larger than 1, got ",
            quantile_);
        CAFFE_ENFORCE_GT(
            tol_, 0, "tolerance should be ", "no less than 0, got ", tol_);
        */
    }

    #[inline] pub fn get_range_from_inputs<T>(
        &mut self,
        lo: *mut T,
        hi: *mut T) 
    {
        todo!();
        /*
            *hi = std::numeric_limits<T>::lowest();
            *lo = T::max;
            for (int i = 0; i < InputSize(); ++i) {
              const auto* input = Input(i).template data<T>();
              for (int j = 0; j < Input(i).numel(); j++) {
                const T val = abs_ ? std::abs(input[j]) : input[j];
                if (*hi < val) {
                  *hi = val;
                }
                if (*lo > val) {
                  *lo = val;
                }
              }
            }
        */
    }

    #[inline] pub fn count_lower_eq<T>(
        &mut self,
        thd: &T) -> i64 {
        todo!();
        /*
            int64_t count = 0;
            for (int i = 0; i < InputSize(); ++i) {
              const auto* input = Input(i).template data<T>();
              for (int j = 0; j < Input(i).numel(); j++) {
                const T val = abs_ ? std::abs(input[j]) : input[j];
                if (val <= thd) {
                  count++;
                }
              }
            }
            return count;
        */
    }
}
