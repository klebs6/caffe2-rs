/*!
  | This file provides two functions to help write
  | elementwise kernels:
  |
  |   cpu_kernel(TensorIterator iter, <lambda>)
  |
  |   cpu_kernel_vec(TensorIterator iter, <lambda>, <vec_lambda>)
  |
  | Both functions may generate vectorized
  | code. The cpu_kernel implementation relies on
  | the compiler's auto-vectorization. The
  | cpu_kernel_vec implementation uses x86 SIMD
  | intrinsics when available. These functions are
  | only intended to be used in the ATen/native/cpu
  | subdirectory, since files in other directories
  | are not compiled with AVX/AVX2 enabled. See
  | README.md for more details.
  |
  | For example, to write a multiplication kernel
  | for float:
  |
  |   cpu_kernel(iter, [](float a, float b) { return a * b; });
  |
  | Or you may write:
  |
  |   cpu_kernel_vec(iter,
  |     [](float a, float b) { return a * b; },
  |     [](Vectorized<float> a, Vectorized<float> b) { return a * b; });
  |
  | See BinaryOpsKernel.cpp for the complete
  | implementation
  |
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/Loops.h]

pub fn dereference_impl<traits, const INDEX: usize>(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64,
        _3:      IndexSequence<INDEX>) -> traits::ArgsTuple {

    todo!();
        /*
            return make_tuple(
          *(typename traits::template arg<INDEX>::type*)
            (data[INDEX] + i * strides[INDEX])...);
        */
}

pub fn dereference<traits>(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64) -> traits::ArgsTuple {

    todo!();
        /*
            using Indices = make_index_sequence<traits::arity>;
      return dereference_impl<traits>(data, strides, i, Indices{});
        */
}

pub fn dereference_vec_impl<traits, const INDEX: usize>(
        data:       &[*mut u8],
        opt_scalar: &traits::result_type,
        S:          usize,
        i:          i64,
        _4:         IndexSequence<INDEX>) -> traits::ArgsTuple {

    todo!();
        /*
            using Vec = typename traits::result_type;
      using Scalar = typename Vec::value_type;
      return make_tuple(
          S == INDEX + 1 ?
          opt_scalar :
          Vec::loadu(data[INDEX] + i * sizeof(Scalar))...);
        */
}

pub fn dereference_vec<traits>(
        data:       &[*mut u8],
        opt_scalar: &traits::result_type,
        S:          usize,
        i:          i64) -> traits::ArgsTuple {

    todo!();
        /*
            using Indices = make_index_sequence<traits::arity>;
      return dereference_vec_impl<traits>(data, opt_scalar, S, i, Indices{});
        */
}

#[inline] pub fn execute_op(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64,
        n:       i64,
        op:      Func)  {

    match op.result_type_is_not_void() {

        //template <typename func_t, typename enable_if<!is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
        true => {

            todo!();
                /*
                    using traits = function_traits<func_t>;
              using result_type = typename traits::result_type;
              for (; i < n; i++) {
                result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
                *out_ptr = apply(forward<func_t>(op), dereference<traits>(
                    &data[1],
                    &strides[1],
                    i));
              }
                */

        }

        //template <typename func_t, typename enable_if<is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
        false => {
            todo!();
                /*
              using traits = function_traits<func_t>;
              for (; i < n; i++) {
                apply(forward<func_t>(op), dereference<traits>(
                    &data[0],
                    &strides[0],
                    i));
              }
                */

        }
    }
    
}

/**
  | Basic loop operation (one output,
  | N inputs). May be auto-vectorized by the
  | compiler. Supports inputs and outputs of
  | different types.
  */
#[inline] pub fn basic_loop<func_t>(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64,
        n:       i64,
        op:      Func)  {

    todo!();
        /*
            using traits = function_traits<func_t>;
      constexpr int ntensors = traits::arity + 1;

      // Copying strides to temporary array helps auto vectorization in older GCC
      // versions.
      i64 strides[ntensors];
      for (int arg = 0; arg < ntensors; arg++) {
        strides[arg] = strides_[arg];
      }

      execute_op(data, strides, i, n, forward<func_t>(op));
        */
}

/**
  | the recursive variadic template for
  | iterating over the returned tuple
  |
  */
pub struct TupleOutput<T,const N: usize> {

}

impl<T,const N: usize> TupleOutput<T,N> {
    
    pub fn handle(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64,
        tuple:   &T)  {
        
        todo!();
        /*
            TupleOutput<T, N - 1>::handle(data, strides, i, tuple);

        auto output = get<N - 1>(tuple);
        using output_type = decltype(output);
        output_type * out_ptr = (output_type *)(data[N - 1] + i * strides[N - 1]);
        *out_ptr = output;
        */
    }
}

/**
  | Base case for the above recursive template
  |
  | template<class T> struct TupleOutput<T, 1> {
  |
  */
pub struct TupleOutputBaseCase {

}

impl TupleOutputBaseCase {
    
    pub fn handle(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64,
        tuple:   &T)  {
        
        todo!();
        /*
            auto output = get<0>(tuple);
        using output_type = decltype(output);
        output_type* out_ptr = (output_type *)(data[0] + i * strides[0]);
        *out_ptr = output;
        */
    }
}

pub fn handle_tuple_outputs<Args>(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64,
        tuple:   &(Args))  {

    todo!();
        /*
            TupleOutput<decltype(tuple), sizeof...(Args)>::handle(data, strides, i, tuple);
        */
}

/**
  | Loop operation for
  | `cpu_kernel_multiple_outputs`.
  |
  | 1. Use `apply` to make dynamic method
  |    invocation for the lambda passed in
  |    `cpu_kernel_multiple_outputs`.
  |
  | 2. Iterate over the members of the returned
  |    tuple, set the corresponding output tensor
  |    by the tuple member in
  |    `handle_tuple_outputs` function.
  |
  */
#[inline] pub fn multiple_outputs_loop<func_t>(
        data:    &[*mut u8],
        strides: *const i64,
        i:       i64,
        n:       i64,
        op:      Func)  {

    todo!();
        /*
            using traits = function_traits<func_t>;

      using result_type = typename traits::result_type;
      constexpr int num_outputs = tuple_size<result_type>::value;
      constexpr int ntensors = traits::arity + num_outputs;

      // Copying strides to temporary array helps auto vectorization in older GCC
      // versions.
      i64 strides[ntensors];
      for (int arg = 0; arg < ntensors; arg++) {
        strides[arg] = strides_[arg];
      }

      for (; i < n; i++) {
        auto output = apply(op, dereference<traits>(
          &data[num_outputs],
          &strides[num_outputs],
          i));
        handle_tuple_outputs(data, strides, i, output);
      }
        */
}


/**
  | Explicitly vectorized loop implementation.
  |
  | All inputs and outputs must be the same type
  | and contiguous with one exception: a single
  | input may be a scalar (stride 0).
  |
  | It's position is indicated by the argument
  | `S`. If `S` is 0, then there are no scalar
  | inputs.
  |
  */
#[inline] pub fn vectorized_loop<func_t, vec_func_t>(
        data: *mut *mut u8,
        n:    i64,
        S:    i64,
        op:   Func,
        vop:  VecFunc)  {

    todo!();
        /*
            using traits = function_traits<vec_func_t>;
      using Scalar = typename function_traits<func_t>::result_type;
      using Vec = Vectorized<Scalar>;
      constexpr int ntensors = traits::arity + 1;

      char*  data[ntensors];
      for (int arg = 0; arg < ntensors; arg++) {
        data[arg] = data_[arg];
      }

      Vec opt_scalar = Vec(S > 0 ? *(Scalar*)data[S] : Scalar(0));
      i64 i = 0;
      for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
        auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
        auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
        auto out1 = apply(forward<vec_func_t>(vop), move(args1));
        auto out2 = apply(forward<vec_func_t>(vop), move(args2));
        out1.store(data[0] + i * sizeof(Scalar));
        out2.store(data[0] + (i + Vec::size()) * sizeof(Scalar));
      }
      if (i < n) {
        i64 strides[ntensors];
        for (int arg = 0; arg < ntensors; arg++) {
          strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(Scalar);
        }
        basic_loop(data, strides, i, n, forward<func_t>(op));
      }
        */
}

#[inline] pub fn unroll_contiguous_scalar_checks<traits, cb_t, const INDEX0: usize, const INDEX: usize>(
    strides: *const i64,
    _1:      IndexSequence<INDEX0,INDEX>,
    cb:      Cb)  {

    /// base case
    /*
    #[inline] pub fn unroll_contiguous_scalar_checks<traits, cb_t>(
            strides: *const i64,
            _1:      IndexSequence,
            cb:      Cb)  {

        todo!();
            /*
                cb(0);
            */
    }
    */

    todo!();
        /*
            if (is_contiguous_scalar<traits, INDEX0 + 1>(strides)) {
        cb(INDEX0 + 1);
      } else {
        unroll_contiguous_scalar_checks<traits>(strides, index_sequence<INDEX...>{}, forward<cb_t>(cb));
      }
        */
}

pub fn cpu_kernel<func_t>(
        iter:       &mut TensorIteratorBase,
        op:         Func,
        grain_size: i64)  {

    let grain_size: i64 = grain_size.unwrap_or(GRAIN_SIZE);

    todo!();
        /*
            using traits = function_traits<func_t>;
      // this could be extended to work with void return types
      TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
      TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
      // dynamic casting not currently supported on CPU
      TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

      iter.for_each([&](char** data, const i64* strides, i64 n) {
        // basic loop can handle 1d slices with arbitrary strides, and 1d slices is all that
        // iter.for_each is ever sending to the loop lambda
          basic_loop(data, strides, 0, n, forward<func_t>(op));
      }, grain_size);
      iter.cast_outputs();
        */
}


/**
  | This function helps write elementwise kernels
  | that requires multiple outputs.
  |
  | It follows the similar structure of cpu_kernel.
  |
  | Instead of `basic_loop` function, a new
  | `multiple_outputs_loop` function is manipulated
  | to handle multiple return values.
  |
  | For now `needs_dynamic_casting` check is not
  | added as the passed lambda (`func_t`) of
  | `multiple_outputs_loop` returns `tuple` instead
  | of `Scalar`.
  |
  | The `gpu_kernel_multiple_outputs` is also
  | implemented without this check,
  |
  | We could extend `needs_dynamic_casting` to
  | support both `tuple` and `thrust::tuple` in the
  | future.
  */
pub fn cpu_kernel_multiple_outputs<func_t>(
        iter:       &mut TensorIteratorBase,
        op:         Func,
        grain_size: i64)  {
    let grain_size: i64 = grain_size.unwrap_or(GRAIN_SIZE);
    todo!();
        /*
            using traits = function_traits<func_t>;
      TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);

      iter.for_each([&](char** data, const i64* strides, i64 n) {
        multiple_outputs_loop(data, strides, 0, n, forward<func_t>(op));
      }, grain_size);
      iter.cast_outputs();
        */
}

//check_dynamic_cast default == true
//
pub fn cpu_kernel_vec<const check_dynamic_cast: bool, Func, VecFunc>(
    iter:       &mut TensorIteratorBase,
    op:         Func,
    vop:        VecFunc,
    grain_size: i64)  {

    let grain_size: i64 = grain_size.unwrap_or(GRAIN_SIZE);

    todo!();
        /*
            using traits = function_traits<func_t>;
      // this could be extended to work with void return types
      TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
      TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
      // dynamic casting not currently supported on CPU, but some kernels (like Fill)
      // explicitly dynamic_cast, so we give the opt-out of checking.
      if_constexpr<check_dynamic_cast>([&] {
        TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
      });

      iter.for_each([&](char** data, const i64* strides, i64 n) {
        if (is_contiguous<traits>(strides)) {
          vectorized_loop(data, n, 0, forward<func_t>(op), forward<vec_func_t>(vop));
        } else {
          using Indices = make_index_sequence<traits::arity>;
          unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](usize idx) {
            if (idx) {
              vectorized_loop(data, n, idx, forward<func_t>(op), forward<vec_func_t>(vop));
            } else {
              basic_loop(data, strides, 0, n, forward<func_t>(op));
            }
          });
        }
      }, grain_size);
      iter.cast_outputs();
        */
}

pub fn cpu_serial_kernel_with_range<func_t>(
        iter:  &mut TensorIteratorBase,
        op:    Func,
        range: &Range)  {

    todo!();
        /*
            using traits = function_traits<func_t>;
      constexpr bool result_void = is_void<typename traits::result_type>::value;
      TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity &&
                            ((result_void && iter.noutputs() == 0) || (!result_void && iter.noutputs() == 1)));
      // dynamic casting not currently supported on CPU
      TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

      iter.serial_for_each([&](char** data, const i64* strides, i64 n) {
        basic_loop(data, strides, 0, n, forward<func_t>(op));
      }, range);
      iter.cast_outputs();
        */
}

pub fn cpu_serial_kernel<func_t>(
        iter: &mut TensorIteratorBase,
        op:   Func)  {

    todo!();
        /*
            cpu_serial_kernel(iter, op, {0, iter.numel()});
        */
}

pub fn cpu_serial_kernel_vec_with_range<func_t, vec_func_t>(
    iter:  &mut TensorIteratorBase,
    op:    Func,
    vop:   VecFunc,
    range: &Range)  {

    todo!();
        /*
            using traits = function_traits<func_t>;
      // this could be extended to work with void return types
      TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
      TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
      // dynamic casting not currently supported on CPU
      TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

      iter.serial_for_each([&](char** data, const i64* strides, i64 n) {
        if (is_contiguous<traits>(strides)) {
          vectorized_loop(data, n, 0, forward<func_t>(op), forward<vec_func_t>(vop));
        } else {
          using Indices = make_index_sequence<traits::arity>;
          unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](usize idx) {
            if (idx) {
              vectorized_loop(data, n, idx, forward<func_t>(op), forward<vec_func_t>(vop));
            } else {
              basic_loop(data, strides, 0, n, forward<func_t>(op));
            }
          });
        }
      }, range);
      iter.cast_outputs();
        */
}

pub fn cpu_serial_kernel_vec<func_t, vec_func_t>(
        iter: &mut TensorIteratorBase,
        op:   Func,
        vop:  VecFunc)  {

    todo!();
        /*
            cpu_serial_kernel_vec(iter, op, vop, {0, iter.numel()});
        */
}
