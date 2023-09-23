/*!
  | The passed in function must take T by value
  | (T), or by
  |
  | const reference (const T&);
  |
  | taking T by non-const reference
  |
  | will result in an error like:
  |
  |    error: no type named 'type' in 'class
  |    result_of<foobar::__lambda(T)>'
  |
  | No explicit template parameters are required.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/functional.h]

/// Overload for explicit function and ArrayRef
///
lazy_static!{
    /*
    template<class F, class T>
    inline auto fmap(const T& inputs, const F& fn) -> vector<decltype(fn(*inputs.begin()))> {
      vector<decltype(fn(*inputs.begin()))> r;
      r.reserve(inputs.size());
      for(const auto & input : inputs)
        r.push_back(fn(input));
      return r;
    }
    */
}

/**
  | C++ forbids taking an address of a constructor,
  | so here's a workaround...
  |
  | Overload for constructor (R) application
  |
  */
#[inline] pub fn fmap<R, T>(inputs: &T) -> Vec<R> {

    todo!();
        /*
            vector<R> r;
      r.reserve(inputs.size());
      for(auto & input : inputs)
        r.push_back(R(input));
      return r;
        */
}

#[inline] pub fn filter<F, T>(
        inputs: &[T],
        fn_:    &F) -> Vec<T> {

    todo!();
        /*
            vector<T> r;
      r.reserve(inputs.size());
      for(auto & input : inputs) {
        if (fn(input)) {
          r.push_back(input);
        }
      }
      return r;
        */
}
