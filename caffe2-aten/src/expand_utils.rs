crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ExpandUtils.h]

/**
  | Named type instead of a pair/tuple so that we
  | can be sure to construct the vectors in place
  | and get NRVO.
  |
  */
pub struct InferExpandGeometryResult<Container> {
    sizes:   Container,
    strides: Container,
}

impl InferExpandGeometryResult<Container> {

    pub fn new(ndim: usize) -> Self {
    
        todo!();
        /*
            : sizes(ndim), strides(ndim)
        */
    }
    
    pub fn new(
        sizes: &[i32],
        ndim:  usize) -> Self {
    
        todo!();
        /*
            : sizes(sizes_.begin(), sizes_.end()), strides(ndim)
        */
    }
}

/**
  | True if input shapes are expandable
  |
  | NOTE: infer_size did a similar check, please
  | keep them sync if change is needed
  */
#[inline] pub fn are_expandable(
        shape1: &[i32],
        shape2: &[i32]) -> bool {
    
    todo!();
        /*
            usize ndim1 = shape1.size();
      usize ndim2 = shape2.size();
      usize ndim = ndim1 < ndim2 ? ndim1 : ndim2;

      for (i64 i = ndim - 1; i >= 0; --i) {
        if (shape1[--ndim1] == shape2[--ndim2] || shape1[ndim1] == 1 || shape2[ndim2] == 1) {
          continue;
        }
        return false;
      }
      return true;
        */
}

/**
  | avoid copy-construction of Tensor
  | by using a reference_wrapper.
  |
  */
#[inline] pub fn check_defined(
        tensors:  InitializerList<ReferenceWrapper<Tensor>>,
        api_name: *const u8)  {
    
    todo!();
        /*
            for (auto& t : tensors) {
        if (!t.get().defined()) {
          AT_ERROR(api_name, "(...) called with an undefined Tensor");
        }
      }
        */
}

/**
  | NOTE [ ExpandUtils Borrowing ]
  |
  | Functions in ExpandUtils return
  | `MaybeOwned<Tensor>` because expansion may not
  | actually be needed, in which case we can
  | improve efficiency by returning
  | `MaybeOwned<Tensor>::borrowed(to_expand)`.
  |
  | However, this means that you need to be
  | careful: the returned `MaybeOwned<Tensor>` must
  | not outlive the original `Tensor` object that
  | `to_expand` referred to! The deleted rvalue
  | reference overloads of these functions help
  | with this by preventing trivial use of
  | a temporary resulting from a function call, but
  | it is still possible to make a mistake.
  */
#[inline] pub fn expand_inplace<'a>(
    tensor:    &Tensor,
    to_expand: &Tensor) -> MaybeOwned<'a,Tensor> {
    
    todo!();
        /*
            if (tensor.sizes().equals(to_expand.sizes())) {
        return MaybeOwned<Tensor>::borrowed(to_expand);
      }
      return MaybeOwned<Tensor>::owned(to_expand.expand(tensor.sizes()));
        */
}

#[inline] pub fn expand_inplace_with_api_name<'a>(
    tensor:    &Tensor,
    to_expand: &Tensor,
    api_name:  *const u8) -> MaybeOwned<'a,Tensor> {
    
    todo!();
        /*
            check_defined({tensor, to_expand}, api_name);
      return expand_inplace(tensor, to_expand);
        */
}

#[inline] pub fn expand_inplace2<'a>(
    tensor:     &Tensor,
    to_expand1: &Tensor,
    to_expand2: &Tensor) -> (MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>) {
    
    todo!();
        /*
            if (tensor.sizes().equals(to_expand1.sizes()) && tensor.sizes().equals((to_expand2.sizes()))) {
        return make_tuple(
            MaybeOwned<Tensor>::borrowed(to_expand1),
            MaybeOwned<Tensor>::borrowed(to_expand2));
      }

      return make_tuple(
          MaybeOwned<Tensor>::owned(to_expand1.expand(tensor.sizes())),
          MaybeOwned<Tensor>::owned(to_expand2.expand(tensor.sizes())));
        */
}

#[inline] pub fn expand_inplace2_with_api_name<'a>(
    tensor:     &Tensor,
    to_expand1: &Tensor,
    to_expand2: &Tensor,
    api_name:   *const u8) -> (MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>) {
    
    todo!();
        /*
            check_defined({tensor, to_expand1, to_expand2}, api_name);
      return expand_inplace(tensor, to_expand1, to_expand2);
        */
}

// See NOTE [ ExpandUtils Borrowing ] above for `MaybeOwned` explanation.
#[inline] pub fn expand_outplace2<'a>(
        to_expand1: &Tensor,
        to_expand2: &Tensor) -> (MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>) {
    
    todo!();
        /*
            if (to_expand1.sizes().equals(to_expand2.sizes())) {
        return make_tuple(
            MaybeOwned<Tensor>::borrowed(to_expand1),
            MaybeOwned<Tensor>::borrowed(to_expand2));
      }

      auto expanded_size = infer_size_dimvector(to_expand1.sizes(), to_expand2.sizes());
      return make_tuple(
          MaybeOwned<Tensor>::owned(to_expand1.expand(expanded_size)),
          MaybeOwned<Tensor>::owned(to_expand2.expand(expanded_size)));
        */
}

#[inline] pub fn expand_outplace_with_api_name<'a>(
    to_expand1: &Tensor,
    to_expand2: &Tensor,
    api_name:   *const u8) -> (MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>) {

    todo!();
        /*
            check_defined({to_expand1, to_expand2}, api_name);
      return expand_outplace(to_expand1, to_expand2);
        */
}

#[inline] pub fn expand_outplace3<'a>(
    to_expand1: &Tensor,
    to_expand2: &Tensor,
    to_expand3: &Tensor) -> (MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>) {

    todo!();
        /*
            if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
        return make_tuple(
            MaybeOwned<Tensor>::borrowed(to_expand1),
            MaybeOwned<Tensor>::borrowed(to_expand2),
            MaybeOwned<Tensor>::borrowed(to_expand3));
      }

      auto expanded_size12 = infer_size_dimvector(to_expand1.sizes(), to_expand2.sizes());
      auto expanded_size = infer_size_dimvector(expanded_size12, to_expand3.sizes());
      return make_tuple(
          MaybeOwned<Tensor>::owned(to_expand1.expand(expanded_size)),
          MaybeOwned<Tensor>::owned(to_expand2.expand(expanded_size)),
          MaybeOwned<Tensor>::owned(to_expand3.expand(expanded_size)));
        */
}

#[inline] pub fn expand_outplace3_with_api_name<'a>(
    to_expand1: &Tensor,
    to_expand2: &Tensor,
    to_expand3: &Tensor,
    api_name:   *const u8) -> (MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>,MaybeOwned<'a,Tensor>) {
    
    todo!();
        /*
            check_defined({to_expand1, to_expand2, to_expand3}, api_name);
      return expand_outplace(to_expand1, to_expand2, to_expand3);
        */
}

#[inline] pub fn expand_size<'a>(
    to_expand: &Tensor,
    sizes:     &[i32]) -> MaybeOwned<'a,Tensor> {
    
    todo!();
        /*
            if (to_expand.sizes().equals(sizes)) {
        return MaybeOwned<Tensor>::borrowed(to_expand);
      }

      return MaybeOwned<Tensor>::owned(to_expand.expand(sizes));
        */
}

#[inline] pub fn expand_size_with_api_name<'a>(
        to_expand: &Tensor,
        sizes:     &[i32],
        api_name:  *const u8) -> MaybeOwned<'a,Tensor> {
    
    todo!();
        /*
            check_defined({to_expand}, api_name);
      return expand_size(to_expand, sizes);
        */
}

#[inline] pub fn expand_outplace(to_expand: &[Tensor]) -> Vec<Tensor> {
    
    todo!();
        /*
            // expands a list of Tensors; ignores undefined (null) tensors
      bool first = true;
      DimVector sizes;
      for (usize i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
          continue;
        } else if (first) {
          sizes = to_expand[i].sizes();
          first = false;
        } else {
          sizes = infer_size_dimvector(sizes, to_expand[i].sizes());
        }
      }

      vector<Tensor> result(to_expand.size());
      for (usize i = 0; i < to_expand.size(); ++i) {
        if (!to_expand[i].defined()) {
          continue;
        } else if (to_expand[i].sizes().equals(sizes)) {
          result[i] = to_expand[i];
        } else {
          result[i] = to_expand[i].expand(sizes);
        }
      }
      return result;
        */
}

/**
  | Sums `tensor` repeatedly to produce a tensor of
  | shape `shape`.
  |
  | Precondition: is_expandable_to(shape,
  | tensor.sizes()) must be true
  */
#[inline] pub fn sum_to(
        tensor: Tensor,
        shape:  &[i32]) -> Tensor {
    
    todo!();
        /*
            if (shape.size() == 0) {
        return tensor.sum();
      }
      SmallVector<i64, 8> reduce_dims;
      const IntArrayRef sizes = tensor.sizes();
      const i64 leading_dims = sizes.size() - shape.size();
      for (i64 i = 0; i < leading_dims; ++i) {
        reduce_dims.push_back(i);
      }
      for (i64 i = leading_dims; i < static_cast<i64>(sizes.size()); ++i) {
        if (shape[i - leading_dims] == 1 && sizes[i] != 1) {
          reduce_dims.push_back(i);
        }
      }
      if (!reduce_dims.empty()) {
        tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
      }
      return leading_dims > 0 ? tensor.view(shape) : tensor;
        */
}

// True if `shape` can be broadcasted to `desired`
#[inline] pub fn is_expandable_to(
        shape:   &[i32],
        desired: &[i32]) -> bool {
    
    todo!();
        /*
            usize ndim = shape.size();
      usize target_dim = desired.size();
      if (ndim > target_dim) {
        return false;
      }
      for (usize i = 0; i < ndim; i++) {
        i64 size = shape[ndim - i - 1];
        i64 target = desired[target_dim - i - 1];
        if (size != target && size != 1) {
          return false;
        }
      }
      return true;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ExpandUtils.cpp]

/**
  | -----------
  | @note
  | 
  | are_expandable did a similar check,
  | please keep them sync if change is needed
  |
  */
pub fn infer_size_impl<Container>(
        a: &[i32],
        b: &[i32]) -> Container {

    todo!();
        /*
            usize dimsA = a.size();
      usize dimsB = b.size();
      usize ndim = dimsA > dimsB ? dimsA : dimsB;
      Container expandedSizes(ndim);

      // Use ptrdiff_t to ensure signed comparison.
      for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
        ptrdiff_t offset = ndim - 1 - i;
        ptrdiff_t dimA = dimsA - 1 - offset;
        ptrdiff_t dimB = dimsB - 1 - offset;
        i64 sizeA = (dimA >= 0) ? a[dimA] : 1;
        i64 sizeB = (dimB >= 0) ? b[dimB] : 1;

        TORCH_CHECK(
            sizeA == sizeB || sizeA == 1 || sizeB == 1,
            "The size of tensor a (", sizeA,
            ") must match the size of tensor b (", sizeB,
            ") at non-singleton dimension ", i);

          // 1s map to the other size (even 0).
          expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
      }

      return expandedSizes;
        */
}

pub fn infer_size(
        a: &[i32],
        b: &[i32]) -> Vec<i64> {
    
    todo!();
        /*
            return infer_size_impl<vector<i64>>(a, b);
        */
}

pub fn infer_size_dimvector(
        a: &[i32],
        b: &[i32]) -> DimVector {
    
    todo!();
        /*
            return infer_size_impl<DimVector>(a, b);
        */
}

pub fn infer_expand_geometry_impl<Container>(
        tensor_sizes:   &[i32],
        tensor_strides: &[i32],
        sizes:          &[i32]) -> InferExpandGeometryResult<Container> {

    todo!();
        /*
            i64 ndim = sizes.size();
      i64 tensor_dim = tensor_sizes.size();

      if (tensor_dim == 0) {
        return InferExpandGeometryResult<Container>(sizes, ndim);
      }

      InferExpandGeometryResult<Container> result(ndim);
      auto& expandedSizes = result.sizes;
      auto& expandedStrides = result.strides;

      // create a new geometry for the tensors
      for (i64 i = ndim - 1; i >= 0; --i) {
        i64 offset = ndim - 1 - i;
        i64 dim = tensor_dim - 1 - offset;
        i64 size = (dim >= 0) ? tensor_sizes[dim] : 1;
        i64 stride = (dim >= 0) ? tensor_strides[dim]
                                    : expandedSizes[i + 1] * expandedStrides[i + 1];
        i64 targetSize = sizes[i];
        if (targetSize == -1) {
          TORCH_CHECK(
              dim >= 0,
              "The expanded size of the tensor (",
              targetSize,
              ") isn't allowed in a leading, non-existing dimension ",
              i);
          targetSize = size;
        }
        if (size != targetSize) {
          TORCH_CHECK(
              size == 1,
              "The expanded size of the tensor (",
              targetSize,
              ") must match the existing size (",
              size,
              ") at non-singleton dimension ",
              i,
              ".  Target sizes: ",
              sizes,
              ".  Tensor sizes: ",
              tensor_sizes);
          size = targetSize;
          stride = 0;
        }
        expandedSizes[i] = size;
        expandedStrides[i] = stride;
      }
      return result;
        */
}

pub fn infer_expand_geometry(
        tensor_sizes:   &[i32],
        tensor_strides: &[i32],
        sizes:          &[i32]) -> (Vec<i64>,Vec<i64>) {
    
    todo!();
        /*
            auto result = inferExpandGeometryImpl<vector<i64>>(
          tensor_sizes, tensor_strides, sizes);
      return make_tuple(move(result.sizes), move(result.strides));
        */
}

pub fn infer_expand_geometry_dimvector(
        tensor_sizes:   &[i32],
        tensor_strides: &[i32],
        sizes:          &[i32]) -> InferExpandGeometryResult<DimVector> {
    
    todo!();
        /*
            return inferExpandGeometryImpl<DimVector>(
          tensor_sizes, tensor_strides, sizes);
        */
}

/**
  | This function returns a dense and
  | non-overlapping strides, which keeps the same
  | layout permutation as the input
  | `tensor_strides`, computed based on the input
  | `tensor_sizes`.
  |
  | Note:
  |
  | 1. This function expects the inputs
  | `tensor_strides` and `tensor_sizes` are
  | non-dense or overlapping,
  |
  |    If the inputs are densed and
  |    non-overlapping, the output strides will be
  |    the same as `tensor_strides`.
  |
  |    However, this function won't check whether
  |    inputs are dense or overlapping, so the
  |    whole function will still be executed even
  |    the inputs are already dense and
  |    non-overlapping, this will cause slowness.
  |
  |    Please verify whether the inputs are
  |    non-dense or overlapping before calling this
  |    function if possible, if the inputs come
  |    from a tensor, you can check this through
  |    `is_non_overlapping_and_dense()`
  |
  | 2. The strides propagation rule that is used in
  |    this function is exactily the same as what
  |    is being used in TensorIterator. Please
  |    refer to
  |    https://github.com/pytorch/pytorch/pull/42922
  |    for more details
  */
pub fn infer_dense_strides(
        tensor_sizes:   &[i32],
        tensor_strides: &[i32]) -> Vec<i64> {
    
    todo!();
        /*
            TORCH_CHECK(tensor_sizes.size() == tensor_strides.size(),
        "Input sizes and strides should have same size but got ", tensor_sizes.size(), " and ", tensor_strides.size());

      usize ndim = tensor_sizes.size();
      if (ndim == 0) {
        return {};
      }
      if (ndim == 1) {
        return {1};
      }

      vector<i64> perm(ndim);
      // initialize perm with n-1, n-2, ..., 1, 0
      iota(perm.rbegin(), perm.rend(), 0);

      // The following sorting algorithm has exactly the same behavior as TensorIterator
      // This is to make sure we have the same stride propagation everywhere.

      // return -1 if dim0 should come before dim1
      // return  1 if dim0 should come after dim1
      // return  0 if comparison is ambiguous
      auto should_swap = [&](usize dim0, usize dim1) {
        i64 stride0 = tensor_strides[dim0];
        i64 stride1 = tensor_strides[dim1];

        // if any stride is 0, treat it as ambiguous comparison to
        // keep the same behavior as TensorIterator
        if (stride0 == 0 || stride1 == 0) {
          return 0;
        }
        if (stride0 < stride1) {
          return -1;
        }
        if (stride0 > stride1) {
          return 1;
        }
        // for equal strides, the dimension with smaller size goes front
        if (tensor_sizes[dim0] > tensor_sizes[dim1]) {
          return 1;
        }
        return 0;
      };

      // Insertion sort (stable) indices in `perm` based on input tensor's stride and shape,
      // all dimensions with 0 stride won't move. This is the same behavior as TensorIterator.
      // eg. Given tensor with size/stride (6, 5, 4, 3, 2)/(6, 0, 120, 0, 1), the initial `perm`
      //     is (4, 3, 2, 1, 0) and the sorted `perm` will be (4, 3, 0, 1, 2)
      for (const auto i : irange(1, ndim)) {
        auto dim1 = i;
        for (const auto j : irange(1, i + 1)) {
          auto dim0 = i - j;
          int comparison = should_swap(perm[dim0], perm[dim1]);
          if (comparison > 0) {
            swap(perm[dim0], perm[dim1]);
            dim1 = dim0;
          }
          else if (comparison < 0) {
            break;
          }
        }
      }

      // compute output strides which preserves the input tensor's memory layout
      vector<i64> out_strides(ndim);
      i64 curr_stride = 1;
      for (usize i = 0; i < ndim; ++i) {
        i64 idx = perm[i];
        out_strides[idx] = curr_stride;
        // Note: for size 0, we simply treated it as 1, it really doesn't matter here
        // since the total number of element is 0.
        if (tensor_sizes[idx] > 1) {
          curr_stride *= tensor_sizes[idx];
        }
      }
      return out_strides;
        */
}

