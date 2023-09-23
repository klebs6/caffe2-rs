crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/CPUApplyUtils.h]

/**
  | [collapse dims] Updates sizes, and
  | strides to reflect a "collapse" of the
  | info, possibly excluding the optional
  | excludeDim. A "collapsed" version
  | of the info is the fewest dims that order
  | the tensor's elements in the same way
  | as the original info. If excludeDim
  | is specified, the collapse is the fewest
  | dims that order the tensor's elements
  | as the original and preserve the excluded
  | dimension, unless the tensor collapses
  | to a point.
  | 
  | This function returns a pair of values.
  | 
  | 1) The (new) index of the preserved dimension
  | if excludeDim is specified. 0 if the
  | tensor is collapsed to a point. -1 otherwise.
  | 
  | 2) The new number of dimensions.
  |
  */
#[inline] pub fn collapse_dims<T>(
    sizes:       *mut T,
    strides:     *mut T,
    dims:        i64,
    exclude_dim: i32) -> (i64,i64) {

    let exclude_dim: i32 = exclude_dim.unwrap_or(-1);
    todo!();
        /*
            TORCH_CHECK(
          excludeDim >= -1 && excludeDim < dims,
          "expected excluded dim between -1 and dims - 1");

      i64 stopDim = (excludeDim == -1) ? dims : excludeDim;
      i64 newIndex = -1;
      i64 oldIndex = 0;
      i64 remappedExcludedDim = -1;

      while (oldIndex < dims) {
        // Finds a dimension to collapse into
        for (; oldIndex < stopDim; ++oldIndex) {
          if (sizes[oldIndex] == 1) {
            continue;
          }

          ++newIndex;
          sizes[newIndex] = sizes[oldIndex];
          strides[newIndex] = strides[oldIndex];
          ++oldIndex;
          break;
        }

        // Collapses dims
        for (; oldIndex < stopDim; ++oldIndex) {
          if (sizes[oldIndex] == 1) {
            continue;
          }

          if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex]) {
            sizes[newIndex] *= sizes[oldIndex];
            strides[newIndex] = strides[oldIndex];
          } else {
            ++newIndex;
            sizes[newIndex] = sizes[oldIndex];
            strides[newIndex] = strides[oldIndex];
          }
        }

        // Handles excludeDim being set (oldIndex == excludeDim)
        if (oldIndex != dims) {
          // Preserves excluded dimension
          ++newIndex;
          sizes[newIndex] = sizes[oldIndex];
          strides[newIndex] = strides[oldIndex];
          remappedExcludedDim = newIndex;

          // Restarts iteration after excludeDim
          ++oldIndex;
          stopDim = dims;
        }
      }

      // Handles special case of all dims size 1
      if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
        dims = 1;
        sizes[0] = 1;
        strides[0] = 1;

        return std::pair<i64, i64>(0, 1);
      }

      dims = newIndex + 1;
      return std::pair<i64, i64>(remappedExcludedDim, dims);
        */
}

/**
  | The basic strategy for apply is as follows:
  | 1. Starting with the outermost index,
  | loop until we reach a dimension where
  | the data is no longer contiguous, i.e.
  | the stride at that dimension is not equal
  | to the size of the tensor defined by the
  | outer dimensions. Let's call this outer
  | (contiguous) tensor A. Note that if
  | the Tensor is contiguous, then
  | 
  | A is equal to the entire Tensor. Let's
  | call the inner tensor B. 2. We loop through
  | the indices in B, starting at its outermost
  | dimension. For example, if B is a 2x2
  | matrix, then we do:
  | 
  | B[0][0]
  | 
  | B[0][1]
  | 
  | B[1][0]
  | 
  | B[1][1]
  | 
  | We set the offset into the underlying
  | storage as (storageOffset + stride_B
  | * index_B), i.e. basically we compute
  | the offset into the storage as we would
  | normally for a Tensor. But because we
  | are guaranteed the subsequent data
  | is contiguous in memory, we can simply
  | loop for sizeof(A) iterations and perform
  | the operation, without having to follow
  | the order described by the strides of
  | 
  | A. 3. As an optimization, we merge dimensions
  | of A that are contiguous in memory. For
  | example, if A is a 3x3x3x3 tensor narrowed
  | from a 3x3x4x3 tensor, then the first
  | two dimensions can be merged for the
  | purposes of APPLY, reducing the number
  | of nested loops.
  |
  */
#[inline] pub fn sort_strides(tensor: &mut Tensor) -> Tensor {
    
    todo!();
        /*
            IntArrayRef strides = tensor_.strides();
      std::vector<i64> indices;
      indices.reserve(tensor_.ndimension());
      for (i64 i = 0; i < tensor_.ndimension(); i++) {
        indices.push_back(i);
      }
      std::sort(indices.begin(), indices.end(), [&strides](i64 i1, i64 i2) {
        return strides[i1] > strides[i2];
      });
      Tensor tensor = tensor_.permute(indices);
      return tensor;
        */
}

lazy_static!{
    /*
    template <typename T, int N>
    struct strided_tensor_iter_fixed {
     public:
      T* data_ = NULL;
      i64 dim_ = 0;

      i64 counter_[N] = {0};
      i64 sizes_[N] = {0};
      i64 strides_[N] = {0};

      strided_tensor_iter_fixed(strided_tensor_iter_fixed const&) = delete;
      void operator=(strided_tensor_iter_fixed const& x) = delete;
      strided_tensor_iter_fixed(strided_tensor_iter_fixed&&) = default;
      strided_tensor_iter_fixed(Tensor& tensor, bool sort_strides = false)
          : data_(tensor.data_ptr<T>()) {
        std::memset(counter_, 0, sizeof(i64) * N);
        if (tensor.dim() > 0) {
          std::memcpy(
              sizes_, tensor.sizes().data(), tensor.dim() * sizeof(i64));
          std::memcpy(
              strides_,
              tensor.strides().data(),
              tensor.dim() * sizeof(i64));
        }
        dim_ = std::get<1>(collapse_dims(sizes_, strides_, tensor.ndimension()));
      }
    };

    template <typename T>
    struct strided_tensor_iter {
     private:
     public:
      T* data_ = NULL;
      i64 dim_;

      std::vector<i64> counter_;
      std::vector<i64> sizes_;
      std::vector<i64> strides_;

      strided_tensor_iter(strided_tensor_iter const&) = delete;
      void operator=(strided_tensor_iter const& x) = delete;
      strided_tensor_iter(strided_tensor_iter&&) = default;
      strided_tensor_iter(Tensor& tensor)
          : data_(tensor.data_ptr<T>()),
            dim_(tensor.ndimension()),
            counter_(dim_, 0),
            sizes_(tensor.sizes().vec()),
            strides_(tensor.strides().vec()) {
        dim_ = std::get<1>(collapse_dims(sizes_.data(), strides_.data(), dim_));
      }
    };
    */
}

#[inline] pub fn all_equal_numel(tensors: &[Tensor]) -> bool {
    
    todo!();
        /*
            if (tensors.size() == 0)
        return true;
      i64 all_numel = tensors[0].numel();
      for (usize i = 1; i < tensors.size(); i++) {
        if (tensors[i].numel() != all_numel)
          return false;
      }
      return true;
        */
}

#[inline] pub fn all_equal_numel_error(tensors: &[Tensor]) -> String {
    
    todo!();
        /*
            std::ostringstream oss;
      oss << "inconsistent tensor size, expected ";
      for (usize i = 0; i < tensors.size() - 1; i++) {
        oss << tensors[i].sizes() << ", ";
      }
      oss << "and " << tensors[tensors.size() - 1].sizes()
          << " to have the same number of elements, but got ";
      for (usize i = 0; i < tensors.size() - 1; i++) {
        oss << tensors[i].numel() << ", ";
      }
      oss << "and " << tensors[tensors.size() - 1].numel()
          << " elements respectively";
      return oss.str();
        */
}

#[inline] pub fn apply_preamble(tensors: &[Tensor]) -> bool {
    
    todo!();
        /*
            checkDeviceType("CPU_tensor_apply", tensors, kCPU);
      checkLayout("CPU_tensor_apply", tensors, kStrided);
      if (!_all_equal_numel(tensors))
        AT_ERROR(_all_equal_numel_error(tensors));
      // An empty tensor has no elements
      for (auto& t : tensors)
        if (t.numel() == 0)
          return false;
      return true;
        */
}

#[inline] pub fn max_dim_tensors(tensors: &[Tensor]) -> i64 {
    
    todo!();
        /*
            i64 dim = 0;
      for (auto& t : tensors)
        dim = std::max(dim, t.ndimension());
      return dim;
        */
}

#[inline] pub fn iterate<Arg, Args>(
    size:      i64,
    iter:      &mut Arg,
    iter_tail: &mut Args)  {

    todo!();
        /*
            iter.counter_[iter.dim_ - 1] += size;
      iter.data_ = iter.data_ + size * iter.strides_[iter.dim_ - 1];
      iterate(size, iter_tail...);
        */
}

#[inline] pub fn iterate_continue<Arg, Args>(
    iter:      &mut Arg,
    iter_tail: &mut Args) -> bool {

    /*
    #[inline] pub fn iterate_continue() -> bool {
        
        todo!();
            /*
                return true;
            */
    }
    */

    todo!();
        /*
            return iter.counter_[iter.dim_ - 1] < iter.sizes_[iter.dim_ - 1] &&
          iterate_continue(iter_tail...);
        */
}

#[inline] pub fn max_iterate_size<Arg, Args>(
    iter:      &mut Arg,
    iter_tail: &mut Args) -> i64 {

    /*
    #[inline] pub fn max_iterate_size() -> i64 {
        
        todo!();
            /*
                return i64::max;
            */
    }
    */

    todo!();
        /*
            return std::min(
          (iter.sizes_[iter.dim_ - 1] - iter.counter_[iter.dim_ - 1]),
          max_iterate_size(iter_tail...));
        */
}

#[inline] pub fn iterate_overflow<Arg, Args>(
    iter:      &mut Arg,
    iter_tail: &mut Args)  {

    todo!();
        /*
      if (iter.counter_[iter.dim_ - 1] == iter.sizes_[iter.dim_ - 1]) {
        for (i64 i = iter.dim_ - 1; i > 0; i--) {
          if (iter.counter_[i] == iter.sizes_[i]) {
            iter.counter_[i] = 0;
            iter.counter_[i - 1]++;
            iter.data_ = iter.data_ - (iter.sizes_[i] * iter.strides_[i]) +
                iter.strides_[i - 1];
          }
        }
      }
      iterate_overflow(iter_tail...);
        */
}

#[inline] pub fn forward<Arg, Args>(
    offset:    i64,
    iter:      &mut Arg,
    iter_tail: &mut Args)  {

    todo!();
        /*
            i64 multi = offset;
      for (i64 i = iter.dim_ - 1; i >= 0; i--) {
        i64 inc = multi % iter.sizes_[i];
        multi = multi / iter.sizes_[i];
        iter.data_ = iter.data_ + inc * iter.strides_[i];
        iter.counter_[i] += inc;
      }
      forward(offset, iter_tail...);
        */
}

#[inline] pub fn max_dim<Arg, Args>(
    iter:      &mut Arg,
    iter_tail: &mut Args) -> i64 {

    /*
    #[inline] pub fn max_dim() -> i64 {
        
        todo!();
            /*
                return 0;
            */
    }
    */

    todo!();
        /*
            return std::max(iter.dim_, max_dim(iter_tail...));
        */
}

#[inline] pub fn apply_op<Op, Args>(
    numel:  i64,
    offset: i64,
    op:     &Op,
    iters:  Args)  {

    /*
    #[inline] pub fn apply_op()  {
        
        todo!();
            /*
            
            */
    }
    */

    todo!();
        /*
            // For 0-dim tensors
      if (numel == 1 && max_dim(iters...) == 0) {
        op(*iters.data_...);
        return;
      }
      if (offset > 0)
        forward(offset, iters...);
      // Splitting this into chunks helps the compiler create faster assembly
      for (i64 i = 0; i < numel;) {
        for (; iterate_continue(iters...) && i < numel;) {
          op(*iters.data_...);
          iterate(1, iters...);
          i++;
        }
        iterate_overflow(iters...);
      }
        */
}

/**
  | Apply a pointwise operator to sequence
  | of tensors
  | 
  | The calling convention for op is a function/functor
  | that takes the same number of pointers
  | of type scalar as the number of given
  | tensors.
  | 
  | For example, to compute a = b * c, op would
  | be of the form:
  | 
  | [](scalar* a_val, const scalar* b_val,
  | const scalar* c_val) { a_val[0] = b_val[0]
  | * c_val[0]; };
  |
  */
#[inline] pub fn cpu_tensor_apply2<scalar1, scalar2, Op>(
    tensor1: Tensor,
    tensor2: Tensor,
    op:      Op)  {

    todo!();
        /*
            if (!_apply_preamble({tensor1, tensor2}))
        return;
      if (_max_dim_tensors({tensor1, tensor2}) <= 8) {
        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter_fixed<scalar1, 8>(tensor1),
            strided_tensor_iter_fixed<scalar2, 8>(tensor2));
      } else {
        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter<scalar1>(tensor1),
            strided_tensor_iter<scalar2>(tensor2));
      }
        */
}

#[inline] pub fn cpu_tensor_apply3<scalar1, scalar2, scalar3, Op>(
    tensor1: Tensor,
    tensor2: Tensor,
    tensor3: Tensor,
    op:      Op)  {

    todo!();
        /*
            if (!_apply_preamble({tensor1, tensor2, tensor3}))
        return;
      if (_max_dim_tensors({tensor1, tensor2, tensor3}) <= 8) {
        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter_fixed<scalar1, 8>(tensor1),
            strided_tensor_iter_fixed<scalar2, 8>(tensor2),
            strided_tensor_iter_fixed<scalar3, 8>(tensor3));
      } else {
        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter<scalar1>(tensor1),
            strided_tensor_iter<scalar2>(tensor2),
            strided_tensor_iter<scalar3>(tensor3));
      }
        */
}

#[inline] pub fn cpu_tensor_apply4<scalar1, scalar2, scalar3, scalar4, Op>(
    tensor1: Tensor,
    tensor2: Tensor,
    tensor3: Tensor,
    tensor4: Tensor,
    op:      Op)  {

    todo!();
        /*
            if (!_apply_preamble({tensor1, tensor2, tensor3, tensor4}))
        return;
      if (_max_dim_tensors({tensor1, tensor2, tensor3, tensor4}) <= 8) {
        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter_fixed<scalar1, 8>(tensor1),
            strided_tensor_iter_fixed<scalar2, 8>(tensor2),
            strided_tensor_iter_fixed<scalar3, 8>(tensor3),
            strided_tensor_iter_fixed<scalar4, 8>(tensor4));
      } else {
        apply_op(
            tensor1.numel(),
            0,
            op,
            strided_tensor_iter<scalar1>(tensor1),
            strided_tensor_iter<scalar2>(tensor2),
            strided_tensor_iter<scalar3>(tensor3),
            strided_tensor_iter<scalar4>(tensor4));
      }
        */
}
