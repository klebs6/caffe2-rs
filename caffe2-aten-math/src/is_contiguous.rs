crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/IsContiguous.h]

/**
  | n: number of function arguments (arity)
  |
  | traits: function_traits (see FunctionTraits.h)
  |
  | s: index of scalar argument or -1
  |
  */
//template <int n, int stride_index, typename traits, int s=-1>
pub struct IsContiguous<const N: isize, const STRIDE_INDEX: isize, Traits, const s: isize = -1> {

}

impl IsContiguous {
    
    pub fn eval(strides: *const i64) -> bool {
        
        todo!();
        /*
            using type = typename traits::template arg<n - 1>::type;
        return strides[stride_index] == (s == n ? 0 : sizeof(type)) &&
               IsContiguous<n - 1, stride_index - 1, traits, s>::eval(strides);
        */
    }
}

lazy_static!{
    /*
    // will be called when there is an output exists
    template <typename traits, int s>
    struct IsContiguous<0, 0, traits, s> {
      static bool eval(const i64* strides) {
        return strides[0] == sizeof(typename traits::result_type);
      }
    };

    // will be called when there is no output
    template <typename traits, int s>
    struct IsContiguous<0, -1, traits, s> {
      static bool eval(const i64* strides) {
        return true;
      }
    };
    */
}

// output and all inputs are contiguous
lazy_static!{
    /*
    template <typename traits,
        typename enable_if<is_void<typename traits::result_type>::value>::type* = nullptr>
    static inline bool is_contiguous(const i64* strides) {
      return IsContiguous<traits::arity, traits::arity - 1, traits>::eval(strides);
    }

    template <typename traits,
        typename enable_if<!is_void<typename traits::result_type>::value>::type* = nullptr>
    static inline bool is_contiguous(const i64* strides) {
      return IsContiguous<traits::arity, traits::arity, traits>::eval(strides);
    }
    */
}

// input at `s` is scalar (stride 0); output and
// other inputs are contiguous NB: output is
// typically at strides[0] so first input
// corresponds to s=1
//
lazy_static!{
    /*
    template <typename traits, int s,
        typename enable_if<is_void<typename traits::result_type>::value>::type* = nullptr>
    static inline bool is_contiguous_scalar(const i64* strides) {
      static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
      return IsContiguous<traits::arity, traits::arity - 1, traits, s>::eval(strides);
    }

    template <typename traits, int s,
        typename enable_if<!is_void<typename traits::result_type>::value>::type* = nullptr>
    static inline bool is_contiguous_scalar(const i64* strides) {
      static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
      return IsContiguous<traits::arity, traits::arity, traits, s>::eval(strides);
    }
    */
}