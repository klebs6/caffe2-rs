crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp]

#[inline] pub fn reduce_all_impl_vec<scalar_t, func_t, vec_func_t>(
    output:  &mut Tensor,
    input:   &Tensor,
    ident_v: Scalar,
    op:      Func,
    vop:     VecFunc)  {

    todo!();
        /*
            using Vec = Vectorized<scalar_t>;
      const i64 input_numel = input.numel();
      auto input_data = input.data_ptr<scalar_t>();
      // NOTE: parallel_reduce not support bool type
      scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
        [&](i64 start, i64 end, const scalar_t ident) -> scalar_t {
          scalar_t partial_out = vec::reduce_all<scalar_t>(
            [=](Vec x, Vec y) { return vop(x, y); },
            input_data + start,
            end - start);
          return partial_out;
        }, op);
      output.fill_(result);
        */
}

// For operation not support in avx/avx2
#[inline] pub fn reduce_all_impl<scalar_t, func_t>(
    output:  &mut Tensor,
    input:   &Tensor,
    ident_v: Scalar,
    op:      Func)  {

    todo!();
        /*
            const i64 input_numel = input.numel();
      auto input_data = input.data_ptr<scalar_t>();
      scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
        [&](i64 start, i64 end, const scalar_t ident) -> scalar_t {
          scalar_t partial_out = ident;
          for (i64 i = start; i < end; i++) {
             partial_out = op(partial_out, input_data[i]);
          }
          return partial_out;
        }, op);
      output.fill_(result);
        */
}

pub fn min_all_kernel_impl(
    result: &mut Tensor,
    input:  &Tensor)  {

    todo!();
        /*
            if (input.scalar_type() == ScalarType::Bool) {
        TensorIterator iter = TensorIteratorConfig()
          .add_input(input)
          .build();
        bool result_data  = true;
        cpu_serial_kernel(iter, [&](const bool a) -> void {
          result_data = result_data && a;
        });
        result.fill_(result_data);
      } else if(input.scalar_type() == ScalarType::Long) {
        // for i64, vectorized implementation have performance issue,
        // just use scalar path
        reduce_all_impl<i64>(result, input, upper_bound<i64>(),
          [=](i64 a, i64 b) -> i64 { return min_impl(a, b); });
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "min_all", [&] {
          using Vec = vec::Vectorized<scalar_t>;
          reduce_all_impl_vec<scalar_t>(result, input, upper_bound<scalar_t>(),
            [=] (scalar_t a , scalar_t b) -> scalar_t { return min_impl(a, b); },
            [=](Vec a, Vec b) -> Vec { return minimum(a, b); });
        });
      }
        */
}

pub fn max_all_kernel_impl(
    result: &mut Tensor,
    input:  &Tensor)  {

    todo!();
        /*
            if (input.scalar_type() == ScalarType::Bool) {
        TensorIterator iter = TensorIteratorConfig()
          .add_input(input)
          .build();
        bool result_data  = false;
        cpu_serial_kernel(iter, [&](const bool a) -> void {
          result_data = result_data || a;
        });
        result.fill_(result_data);
      } else if (input.scalar_type() == ScalarType::Long) {
        // for i64, vectorized implementation have performance issue,
        // just use scalar path
        reduce_all_impl<i64>(result, input, lower_bound<i64>(),
          [=](i64 a, i64 b) -> i64 { return max_impl(a, b); });
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "max_all", [&] {
          using Vec = vec::Vectorized<scalar_t>;
          reduce_all_impl_vec<scalar_t>(result, input, lower_bound<scalar_t>(),
            [=] (scalar_t a , scalar_t b) -> scalar_t { return max_impl(a, b); },
            [=](Vec a, Vec b) -> Vec { return maximum(a, b); });
        });
      }
        */
}

// For operation not support in avx/avx2
#[inline] pub fn reduce_all_impl_two_outputs<scalar_t, func_t1, func_t2>(
    output1:           &mut Tensor,
    output2:           &mut Tensor,
    input:             &Tensor,
    ident_v:           &(Scalar,Scalar),
    reduce_chunk_func: FuncT1,
    reduce_acc_func:   FuncT2)  {

    todo!();
        /*
            using scalar_t_pair = std::pair<scalar_t, scalar_t>;
      const i64 input_numel = input.numel();
      auto input_data = input.data_ptr<scalar_t>();
      scalar_t_pair result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
        [&](i64 start, i64 end, const scalar_t_pair& ident) -> scalar_t_pair {
          scalar_t_pair partial_out(ident);
          for (i64 i = start; i < end; i++) {
             partial_out = reduce_chunk_func(partial_out, input_data[i]);
          }
          return partial_out;
        },
        reduce_acc_func
      );
      output1.fill_(result.first);
      output2.fill_(result.second);
        */
}

#[inline] pub fn reduce_all_impl_vec_two_outputs<scalar_t, func_t, vec_func_t1, vec_func_t2>(
    output1:            &mut Tensor,
    output2:            &mut Tensor,
    input:              &Tensor,
    ident_v:            &(Scalar,Scalar),
    reduce_acc_func:    Func,
    reduce_chunk_func1: VecFuncT1,
    reduce_chunk_func2: VecFuncT2)  {

    todo!();
        /*
            using Vec = Vectorized<scalar_t>;
      using scalar_t_pair = std::pair<scalar_t, scalar_t>;
      const i64 input_numel = input.numel();
      auto input_data = input.data_ptr<scalar_t>();
      // NOTE: parallel_reduce not support bool type
      std::pair<scalar_t, scalar_t> result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
        [&](i64 start, i64 end, const scalar_t_pair& /* ident */) -> scalar_t_pair {
        scalar_t_pair partial_out = vec::reduce2_all<scalar_t>(
            [=](Vec x, Vec y) { return reduce_chunk_func1(x, y); },
            [=](Vec x, Vec y) { return reduce_chunk_func2(x, y); },
            input_data + start,
            end - start);
          return partial_out;
        },
        reduce_acc_func
      );
      output1.fill_(result.first);
      output2.fill_(result.second);
        */
}

pub fn aminmax_all_kernel_impl(
    min_result: &mut Tensor,
    max_result: &mut Tensor,
    input:      &Tensor)  {

    todo!();
        /*
            if (input.scalar_type() == ScalarType::Bool) {
        TensorIterator iter = TensorIteratorConfig()
          .add_input(input)
          .build();
        bool min_result_data = true;
        bool max_result_data = false;
        cpu_serial_kernel(iter, [&](const bool a) -> void {
          min_result_data = min_result_data && a;
          max_result_data = max_result_data || a;
        });
        min_result.fill_(min_result_data);
        max_result.fill_(max_result_data);
      } else if (input.scalar_type() == ScalarType::Long) {
        // for i64, vectorized implementation have performance issue,
        // just use scalar path
        using i64_pair = std::pair<i64, i64>;
        reduce_all_impl_two_outputs<i64>(min_result, max_result, input,
          i64_pair(upper_bound<i64>(), lower_bound<i64>()),
          // reduce over chunk
          [=](i64_pair a, i64 b) -> i64_pair {
            return i64_pair(min_impl(a.first, b), max_impl(a.second, b));
          },
          // combine two inputs
          [=](i64_pair a, i64_pair b) -> i64_pair {
            return i64_pair(min_impl(a.first, b.first), max_impl(a.second, b.second));
          }
        );
      } else {
        AT_DISPATCH_ALL_TYPES(input.scalar_type(), "_aminmax_all_all", [&] {
          using Vec = vec::Vectorized<scalar_t>;
          using scalar_t_pair = std::pair<scalar_t, scalar_t>;
          reduce_all_impl_vec_two_outputs<scalar_t>(
            min_result,
            max_result,
            input,
            scalar_t_pair(upper_bound<scalar_t>(), lower_bound<scalar_t>()),
            [=] (scalar_t_pair a , scalar_t_pair b) -> scalar_t_pair {
              return scalar_t_pair(
                min_impl(a.first, b.first), max_impl(a.second, b.second));
            },
            [=](Vec a, Vec b) -> Vec { return minimum(a, b); },
            [=](Vec a, Vec b) -> Vec { return maximum(a, b); }
          );
        });
      }
        */
}

register_dispatch!{
    min_all_stub, 
    &min_all_kernel_impl
}

register_dispatch!{
    max_all_stub, 
    &max_all_kernel_impl
}

register_dispatch!{
    _aminmax_all_stub, 
    &_aminmax_all_kernel_impl
}
