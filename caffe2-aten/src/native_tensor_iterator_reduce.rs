// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorIteratorReduce.cpp]

/// Contains the implementation of parallel reductions in TensorIterator.
pub type Loop2d = TensorIteratorBase::loop2d_t;

impl TensorIteratorBase {
    
    pub fn parallel_reduce(&mut self, loop_: Loop2d)  {
        
        todo!();
        /*
            TORCH_CHECK(ntensors() == 2, "parallel_reduce only supports one input and one output");
      i64 numel = this->numel();
      if (numel < internal::GRAIN_SIZE || get_num_threads() == 1 ||
          in_parallel_region()) {
        serial_for_each(loop, {0, numel});
      } else if (use_two_pass_reduction(*this)) {
        two_pass_reduction(*this, loop);
      } else {
        parallel_dim_reduction(*this, loop);
      }
        */
    }
}

pub fn use_two_pass_reduction(iter: &mut TensorIteratorBase) -> bool {
    
    todo!();
        /*
            return iter.output(0).numel() == 1;
        */
}

pub fn two_pass_reduction(
        iter:  &mut TensorIteratorBase,
        loop_: Loop2d)  {
    
    todo!();
        /*
            const int max_threads = get_num_threads();

      auto dst = iter.output(0);
      auto unsqueezed = dst.unsqueeze(0);
      auto buffer_shape = DimVector(unsqueezed.sizes());
      buffer_shape[0] = max_threads;
      auto buffer = empty(buffer_shape, dst.options());
      // Fill with the identity
      buffer.copy_(unsqueezed);

      auto buffer_stride = buffer.strides()[0] * buffer.element_size();
      auto buffer_0 = buffer[0];
      auto first_reduce = TensorIterator::reduce_op(buffer_0, iter.input(0));
      TORCH_INTERNAL_ASSERT(first_reduce.output(0).is_alias_of(buffer_0));

      parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&](i64 begin, i64 end) {
        const auto thread_num = get_thread_num();
        auto shape = first_reduce.shape();
        auto strides = first_reduce.get_strides();

        // Bump output ptr so each thread has its own ouput slice
        auto base_ptrs = first_reduce.get_base_ptrs();
        base_ptrs[0] += buffer_stride * thread_num;

        internal::serial_for_each(shape, strides, base_ptrs.data(),
                                      base_ptrs.size(), loop, {begin, end});
      });

      auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
      final_reduce.for_each(loop);
        */
}

/**
  | Chooses a dimension over which to
  | parallelize. Prefers the outer-most dimension
  | thats larger than the number of available
  | threads.
  */
pub fn find_split_dim(iter: &mut TensorIteratorBase) -> i32 {
    
    todo!();
        /*
            int num_threads = get_num_threads();
      auto shape = iter.shape();

      // start with the outer-most dimension
      int best_dim = iter.ndim() - 1;
      for (int dim = best_dim; dim >= 0 && !iter.is_dim_reduced(dim); dim--) {
        if (shape[dim] >= num_threads) {
          return dim;
        } else if (shape[dim] > shape[best_dim]) {
          best_dim = dim;
        }
      }

      AT_ASSERT(!iter.is_dim_reduced(best_dim));
      return best_dim;
        */
}

pub fn round_columns(
    iter:     &mut TensorIteratorBase,
    dim:      i32,
    multiple: i32,
    begin:    i64,
    end:      i64) -> (i64,i64) {
    
    todo!();
        /*
            begin = begin - (begin % multiple);
      if (end != iter.shape()[dim]) {
        // only round the 'end' column down if it's not the final column
        end = end - (end % multiple);
      }
      return make_tuple(begin, end);
        */
}

pub fn parallel_dim_reduction(
    iter:  &mut TensorIteratorBase,
    loop_: Loop2d)  {
    
    todo!();
        /*
            AT_ASSERT(iter.ndim() >= 1);
      int dim = find_split_dim(iter);
      i64 cols = iter.shape()[dim];
      int element_size = iter.element_size(/*arg=*/1);

      bool should_round_columns = iter.strides(1)[dim] == element_size;
      parallel_for(0, cols, 1, [&](i64 begin, i64 end) {
        if (should_round_columns) {
          // round columns to multiples of 128 bytes if adjacent columns are
          // contiguous in memory.
          i64 cols_per_128_bytes = 128 / element_size;
          tie(begin, end) = round_columns(iter, dim, cols_per_128_bytes, begin, end);
        }
        if (begin == end) {
          return;
        }
        auto sub_iter = TensorIterator(iter);
        sub_iter.narrow(dim, begin, end - begin);
        sub_iter.for_each(loop);
      });
        */
}

pub fn foreach_reduced_elt(
    loop_:       LoopSubiter,
    parallelize: bool)  {
    
    todo!();
        /*
            AT_ASSERT(ninputs() == 1);
      AT_ASSERT(noutputs() >= 1);

      auto shape = this->shape();
      if (output(0).numel() == 0) {
        return;
      }
      if (output(0).numel() == 1) {
        loop(*this);
      }
      else if (numel() < internal::GRAIN_SIZE || get_num_threads() == 1 ||
          in_parallel_region() || !parallelize) {
        auto reduce_dims = num_reduce_dims();

        auto non_reduced_shape = shape.slice(reduce_dims, shape.size() - reduce_dims);

        i64 non_reduced_numel = 1;
        for (const auto i : irange(non_reduced_shape.size())) {
          non_reduced_numel *= non_reduced_shape[i];
        }
        DimCounter dims {non_reduced_shape, {0, non_reduced_numel}};
        while (!dims.is_done()) {
          TensorIterator reduced = *this;
          reduced.select_all_keeping_dim(reduce_dims, dims.values);
          loop(reduced);
          dims.increment({1, 1});
        }
      }
      else {
        int dim = find_split_dim(*this);
        i64 cols = shape[dim];
        parallel_for(0, cols, 1, [&](i64 begin, i64 end) {
          if (begin == end) {
            return;
          }

          TensorIterator sub_iter(*this);

          sub_iter.narrow(dim, begin, end - begin);
          // On some broken setups, `#ifdef _OPENMP` is true,
          // and `get_num_threads` returns > 1, but
          // `#pragma omp parallel` is ignored.
          // There is no API to check for this, so we need to explicitly
          // stop trying to parallelize if we've already gotten here.
          //
          // (If we are on one of those broken setups, we will
          //  only have one thread here, and end - begin == cols.)
          sub_iter.foreach_reduced_elt(loop, false);
        });
      }
        */
}
