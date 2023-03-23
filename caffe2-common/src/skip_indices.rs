crate::ix!();

pub type CaffeMap<Key,Value> = HashMap<Key,Value>;

/**
  | SkipIndices are used in
  | operator_fallback_gpu.h and
  | operator_fallback_mkl.h as utility functions
  | that marks input / output indices to skip when
  | we use a CPU operator as the fallback of
  | GPU/MKL operator option.
  |
  | note: this is supposed to be a variadic
  | template
  */
pub trait SkipIndices<const V: i32> {

    fn contains_internal(&self, i: i32) -> bool {
        i == V
    }

    /*
      template <int First, int Second, int... Rest>
      static inline bool ContainsInternal(const int i) {
        return (i == First) || ContainsInternal<Second, Rest...>(i);
      }

      static inline bool Contains(const int i) {
        return ContainsInternal<values...>(i);
      }

      static inline bool Contains(const int /*i*/) {
        return false;
      }
    */
}
