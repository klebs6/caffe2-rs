crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/Reduce.h]

#[macro_export] macro_rules! vec_loop_header {
    ($func_t:ident, $data:ident) => {
        /*
        
          using Scalar = typename function_traits<func_t>::result_type; 
          using Vec = Vectorized<Scalar>; 
          char* out_ptr = data[0]; 
          (void) out_ptr;
        */
    }
}

/**
  | reduction that is contiguous over the
  | input in dim 0
  |
  */
#[inline] pub fn is_contiguous_reduction<traits>(strides: *const i64) -> bool {

    todo!();
        /*
            return strides[0] == 0 &&
             strides[1] == sizeof(typename traits::arg2_t);
        */
}

/**
  | reduction that is contiguous over the
  | input in dim 1
  |
  */
#[inline] pub fn is_outer_reduction<traits>(strides: *const i64) -> bool {

    todo!();
        /*
            return strides[0] == 0 &&
             strides[2] == sizeof(typename traits::result_type) &&
             strides[3] == sizeof(typename traits::arg2_t);
        */
}

#[inline] pub fn reduction128<func_t, vec_func_t>(
        data:   *mut *mut u8,
        n:      i64,
        stride: i64,
        op:     Func,
        vop:    VecFunc,
        reduce: bool)  {

    todo!();
        /*
            VEC_LOOP_HEADER(func_t, data)
      const char* in1_ptr = data[1];
      Vec acc[4];
      for  (int j = 0; j < 4; j++) {
        acc[j] = Vec::loadu(in1_ptr + j * Vec::size() * sizeof(Scalar));
      }
      for (i64 i = 1; i < n; i++) {
        const char* ptr = in1_ptr + stride * i;
        acc[0] = vop(acc[0], Vec::loadu(ptr + (0 * Vec::size() * sizeof(Scalar))));
        acc[1] = vop(acc[1], Vec::loadu(ptr + (1 * Vec::size() * sizeof(Scalar))));
        acc[2] = vop(acc[2], Vec::loadu(ptr + (2 * Vec::size() * sizeof(Scalar))));
        acc[3] = vop(acc[3], Vec::loadu(ptr + (3 * Vec::size() * sizeof(Scalar))));
      }
      if (reduce) {
        Scalar buffer[Vec::size()];
        acc[0] = vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]));
        acc[0].store(buffer);
        for (int j = 1; j < Vec::size(); j++) {
          buffer[0] = op(buffer[0], buffer[j]);
        }
        auto dst = (Scalar*)out_ptr;
        *dst = op(*dst, buffer[0]);
      } else {
        for (int j = 0; j < 4; j++) {
          auto dst = out_ptr + j * Vec::size() * sizeof(Scalar);
          acc[j] = vop(acc[j], Vec::loadu(dst));
          acc[j].store(dst);
        }
      }
        */
}

#[inline] pub fn UNARY_OUTER_LOOP<F>(
        data:    [*mut u8; 2],
        strides: [i64; 2],
        n:       i64,
        f:       F)  {

    todo!();
        /*
            for (int j = 0; j < n; j++) {
        f();
        data[0] += strides[0];
        data[1] += strides[1];
      }
        */
}

/**
  | computes the reduction out = op(out,
  | in)
  |
  */
#[inline] pub fn vectorized_inner_reduction<func_t, vec_func_t>(
        data: *mut *mut u8,
        n:    i64,
        op:   Func,
        vop:  VecFunc)  {

    todo!();
        /*
            VEC_LOOP_HEADER(func_t, data)
      i64 vector_stride = 4 * Vec::size() * sizeof(Scalar);
      i64 count = n / (4 * Vec::size());
      if (count > 0) {
        reduction128(data, count, vector_stride, op, vop, /*reduce=*/true);
      }
      char* ptrs[3] = { data[0], data[0], data[1] };
      i64 strides[] = { 0, 0, sizeof(Scalar) };
      basic_loop(ptrs, strides, count * 4 * Vec::size(), n, op);
        */
}

/**
  | computes the reduction out = op(out,
  | in)
  |
  */
#[inline] pub fn vectorized_outer_reduction<func_t, vec_func_t>(
        data:         *mut *mut u8,
        inner_stride: i64,
        size0:        i64,
        size1:        i64,
        op:           Func,
        vop:          VecFunc)  {

    todo!();
        /*
            VEC_LOOP_HEADER(func_t, data)

      // reduce down each column of 4 * Vec::size() elements (128 bytes)
      i64 outer_stride[2] = { 128, 128 };
      UNARY_OUTER_LOOP(data, outer_stride, size1 / (4 * Vec::size()), [&] {
        reduction128(data, size0, inner_stride, op, vop, /*reduce=*/false);
      });

      // reduce down the remaining columns
      i64 step[] = { sizeof(Scalar), sizeof(Scalar) };
      i64 remaining = size1 % (4 * Vec::size());
      UNARY_OUTER_LOOP(data, step, remaining, [&] {
        char* ptrs[3] = { data[0], data[0], data[1] };
        i64 strides[] = { 0, 0, inner_stride };
        basic_loop(ptrs, strides, 0, size0, op);
      });
        */
}

pub fn set_result<traits, res_t>(
        index:       i32,
        result:      Res,
        iter:        &TensorIteratorBase,
        num_outputs: i32)  {

    todo!();
        /*
            // static_assert(is_same<res_t, typename traits::arg2_t>::value, "data types must match");
      if (index < num_outputs) {
        char *out = (char *) iter.data_ptr(index);
        *(res_t *) out = result;
      }
        */
}

lazy_static!{
    /*
    template<typename traits, usize i = 0, typename... tuple_t>
    static inline typename enable_if<i == sizeof...(tuple_t), usize>::type
    for_each_in_tuple(const tuple<tuple_t...>& t, const TensorIteratorBase &iter, const int num_outputs) {
      return i;
    }

    template<typename traits, usize i = 0, typename... tuple_t>
    static inline typename enable_if<i < sizeof...(tuple_t), usize>::type
    for_each_in_tuple(const tuple<tuple_t...>& t, const TensorIteratorBase &iter, const int num_outputs) {
      if (i < (usize)num_outputs) {
        set_result<traits>(i, get<i>(t), iter, num_outputs);
        return for_each_in_tuple<traits, i + 1, tuple_t...>(t, iter, num_outputs);
      }
      return i;
    }
    */
}

pub fn set_results<traits, res_t>(
        result:      Res,
        iter:        &TensorIteratorBase,
        num_outputs: i32)  {

    todo!();
        /*
            AT_ASSERT(num_outputs == 1);
      set_result<traits>(0, result, iter, num_outputs);
        */
}

pub fn set_results_tuple<traits, Res>(
    result:      &(Res),
    iter:        &TensorIteratorBase,
    num_outputs: i32)  {

    todo!();
        /*
            AT_ASSERT(num_outputs >= 1);
      usize result_size = for_each_in_tuple<traits>(result, iter, num_outputs);
      AT_ASSERT((usize)num_outputs == result_size);
        */
}

lazy_static!{
    /*
    template <typename T, typename... Args>
    struct all_same : conjunction<
      is_same<T, Args>...
    > {};
    */
}

/**
  | Data is the input/output data type.
  |
  | Acc is a type that contains all the necessary
  | data to continue reducing.
  |
  | Index is a one-dimensional index
  |
  | ops_t is such that &ops_t::reduce,
  | &ops_t::combine, and &ops_t::project exist and
  | satisfy the following.
  |
  | reduce: (Acc, Data, Index) -> Acc adds
  | one data point to the accumulated value.
  |
  | combine: (Acc, Acc) -> Acc combines two
  | accumulated values into one.
  |
  | project: Acc -> out_t finishes the reduction,
  | getting the required output.
  |
  | Additionally, Acc must be
  | default-constructible:
  |
  | Acc {} is an identity for combine, and
  | project(Acc {}) is the value of the operation
  | on zero elements.
  |
  | The point of `combine` is to support
  | parallelization - the idea is to one sequence
  | of `reduce` calls per thread of execution, and
  | then to combine them at the end with `combine`.
  |
  | If there is more than one output element, our
  | parallelization strategy is to use one thread
  | for each of them, which means that `combine`
  | will never be called.
  |
  | If, on the other hand, there is only one, then
  | we split the input into into several pieces,
  | reduce each separately, and then combine them.
  */
pub fn binary_kernel_reduce<ops_t, init_t>(
        iter: &mut TensorIteratorBase,
        ops:  Ops,
        init: Init)  {

    todo!();
        /*
            using rf_t = decltype(&ops_t::reduce);
      using cf_t = decltype(&ops_t::combine);
      using pf_t = decltype(&ops_t::project);
      using r_traits = binary_function_traits<rf_t>;
      using c_traits = binary_function_traits<cf_t>;
      using p_traits = unary_function_traits<pf_t>;
      using Acc = typename p_traits::arg1_t;
      using Data = typename r_traits::arg2_t;
      static_assert(
        all_same<
          Acc,
          init_t,
          typename r_traits::arg1_t,
          typename r_traits::result_type,
          typename c_traits::arg1_t,
          typename c_traits::arg2_t,
          typename c_traits::result_type>::value,
        "all accumulate types must match");
      static_assert(
        is_default_constructible<Acc>::value,
        "the accumulate type must be default-constructible"
      );
      const int num_outputs = iter.noutputs();
      iter.foreach_reduced_elt([&ops, &init, num_outputs](TensorIteratorBase &sub_iter) {
        auto reduction_body = [&ops, &sub_iter, num_outputs](Acc acc, i64 begin, i64 end) -> Acc {
          int ntensors = sub_iter.ntensors();
          sub_iter.serial_for_each([&acc, &ops, num_outputs, ntensors, begin](char** data, const i64* strides, i64 size) {
            AT_ASSERT(ntensors - num_outputs == 1);
            char *in = data[ntensors - 1];
            i64 stride = strides[ntensors - 1];
            for (i64 i = 0; i < size; ++i) {
              acc = ops.reduce(acc, *(Data*)in, begin + i);
              in += stride;
            }
          }, {begin, end});
          return ops.translate_idx(acc, sub_iter.view_offsets()[0]);
        };
        Acc total_acc = init;
        auto numel = sub_iter.numel();
        if (numel < internal::GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
          total_acc = reduction_body(total_acc, 0, numel);
        } else {
          int max_threads = get_num_threads();
          AT_ASSERT(max_threads > 0);
          static_assert(
            !is_same<Acc, bool>::value,
            "Concurrently modifying different references into vector<bool> is UB."
          );
          vector<Acc> buffer((unsigned)max_threads, init);
          parallel_for(0, numel, internal::GRAIN_SIZE,
            [&](i64 begin, i64 end) {
              auto& acc = buffer[get_thread_num()];
              acc = reduction_body(acc, begin, end);
            }
          );
          for (int i = 0; i < max_threads; ++i) {
            total_acc = ops.combine(total_acc, buffer[i]);
          }
        }
        set_results<r_traits>(ops.project(total_acc), sub_iter, num_outputs);
      });
        */
}

pub fn binary_kernel_reduce_vec<func_t, vec_func_t>(
        iter:  &mut TensorIteratorBase,
        op:    Func,
        vop:   VecFunc,
        ident: f64)  {
    let ident: f64 = ident.unwrap_or(0);
    todo!();
        /*
            using traits = binary_function_traits<func_t>;
      static_assert(
        all_same<
          typename traits::result_type,
          typename traits::arg1_t,
          typename traits::arg2_t>::value,
        "all types must match");

      iter.output().fill_(ident);
      iter.parallel_reduce([&](char** data, const i64* strides, i64 size0, i64 size1) {
        i64 outer_strides[] = { strides[2], strides[3] };
        if (is_contiguous_reduction<traits>(strides)) {
          // input is contiguous in dim 0, output is reduced in dim 0
          UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
            vectorized_inner_reduction(data, size0, op, vop);
          });
        } else if (is_outer_reduction<traits>(strides)) {
          // input and output are contiguous in dim 1
          i64 inner_stride = strides[1]; // stride of input in dim 0
          vectorized_outer_reduction(data, inner_stride, size0, size1, op, vop);
        } else {
          UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
            char* ptrs[3] = { data[0], data[0], data[1] };
            i64 inner_strides[3] = { strides[0], strides[0], strides[1] };
            basic_loop(ptrs, inner_strides, 0, size0, op);
          });
        }
      });
        */
}
