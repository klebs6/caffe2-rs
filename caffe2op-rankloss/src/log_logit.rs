crate::ix!();

/**
  | Computes log(1 + exp(y)) in a way that
  | avoids early over-/under-flow
  |
  */
#[inline] pub fn log_logit<T>(x: T) -> T {

    todo!();
    /*
        static const auto kMinLogDiff = std::log(std::numeric_limits<T>::epsilon());

      if (x < kMinLogDiff) {
        return 0;
      }
      if (x > -kMinLogDiff) {
        return x;
      }
      return std::log(std::exp(x) + 1);
    */
}
