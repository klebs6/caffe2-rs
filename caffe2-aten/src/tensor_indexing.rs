crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorIndexing.h]

/**
  | TODO: try to remove this
  |
  | There is some back story, see
  | https://github.com/pytorch/pytorch/issues/48684
  |
  */
pub const INDEX_MAX: i64 = i64::MAX;
pub const INDEX_MIN: i64 = i64::MIN;

pub enum TensorIndexType { 
    None, 
    Ellipsis, 
    Integer, 
    Boolean, 
    Slice, 
    Tensor 
}

pub const NONE: Option<()> = None;

#[derive(Default)]
pub struct EllipsisIndexType {

}

lazy_static!{
    /*
    extern const EllipsisIndexType Ellipsis;
    */
}

pub struct Slice {
    start: i64,
    stop:  i64,
    step:  i64,
}

impl Slice {

    /// This mirrors `__PySlice_Unpack` in
    /// torch/csrc/utils/python_compat.h
    ///
    pub fn new(
        start_index: Option<i64>,
        stop_index:  Option<i64>,
        step_index:  Option<i64>) -> Self {

        todo!();
        /*


            if (!step_index.has_value()) {
          step_ = 1;
        } else {
          step_ = step_index.value();
          TORCH_CHECK_VALUE(step_ != 0, "slice step cannot be zero");

          // Here step might be -INDEX_MAX-1; in this case we replace it
          // with -INDEX_MAX.  This doesn't affect the semantics, and it
          // guards against later undefined behaviour resulting from code that
          // does "step = -step" as part of a slice reversal.
          if (step_ < -INDEX_MAX)
            step_ = -INDEX_MAX;
        }
        if (!start_index.has_value()) {
          start_ = step_ < 0 ? INDEX_MAX : 0;
        } else {
          start_ = start_index.value();
        }
        if (!stop_index.has_value()) {
          stop_ = step_ < 0 ? INDEX_MIN : INDEX_MAX;
        } else {
          stop_ = stop_index.value();
        }
        */
    }
    
    #[inline] pub fn start(&self) -> i64 {
        
        todo!();
        /*
            return start_;
        */
    }
    
    #[inline] pub fn stop(&self) -> i64 {
        
        todo!();
        /*
            return stop_;
        */
    }
    
    #[inline] pub fn step(&self) -> i64 {
        
        todo!();
        /*
            return step_;
        */
    }
}

/**
  | `TensorIndex` is used for converting
  | C++ tensor indices such as
  |
  | `{None, "...", Ellipsis, 0, true, Slice(1,
  | None, 2), Torchtensor({1, 2})}`
  |
  | into its equivalent `vector<TensorIndex>`, so
  | that further tensor indexing operations can be
  | performed using the supplied indices.
  |
  | There is one-to-one correspondence between
  | Python and C++ tensor index types:
  |
  | Python                  | C++
  | -----------------------------------------------------
  | `None`                  | `None`
  | `Ellipsis`              | `Ellipsis`
  | `...`                   | `"..."`
  | `123`                   | `123`
  | `True` / `False`        | `true` / `false`
  | `:`                     | `Slice()` / `Slice(None, None)`
  | `::`                    | `Slice()` / `Slice(None, None, None)`
  | `1:`                    | `Slice(1, None)`
  | `1::`                   | `Slice(1, None, None)`
  | `:3`                    | `Slice(None, 3)`
  | `:3:`                   | `Slice(None, 3, None)`
  | `::2`                   | `Slice(None, None, 2)`
  | `1:3`                   | `Slice(1, 3)`
  | `1::2`                  | `Slice(1, None, 2)`
  | `:3:2`                  | `Slice(None, 3, 2)`
  | `1:3:2`                 | `Slice(1, 3, 2)`
  | `torch.tensor([1, 2])`) | `Torchtensor({1, 2})`
  */
pub struct TensorIndex {
    integer: i64,
    boolean: bool,
    slice:   Slice,
    tensor:  Tensor,
    ty:      TensorIndexType,
}

impl Default for TensorIndex {

    /**
      | Case 1: `None`
      | 
      |
      */
    fn default() -> Self {
    
        todo!();
        /*
        : ty(TensorIndexType::None),

        
        */
    }
}

impl From<EllipsisIndexType> for TensorIndex {

    /**
      | Case 2: "..." / `Ellipsis`
      |
      */
    fn from(_0: EllipsisIndexType) -> Self {
    
        todo!();
        /*
        : ty(TensorIndexType::Ellipsis),

        
        */
    }
}

impl From<*const u8> for TensorIndex {

    fn from(str_: *const u8) -> Self {
    
        todo!();
        /*
        : tensor_index(Ellipsis),

        TORCH_CHECK_VALUE(
          strcmp(str, "...") == 0,
          "Expected \"...\" to represent an ellipsis index, but got \"", str, "\"");
        */
    }
}

impl From<i64> for TensorIndex {

    /**
      | Case 3: Integer value
      | 
      |
      */
    fn from(integer: i64) -> Self {
    
        todo!();
        /*
        : integer(integer),
        : ty(TensorIndexType::Integer),

        
        */
    }
}

impl From<i32> for TensorIndex {

    fn from(integer: i32) -> Self {
    
        todo!();
        /*


            : TensorIndex((i64)integer)
        */
    }
}

impl From<bool> for TensorIndex {

    // Case 4: Boolean value
    fn from(boolean: bool) -> Self {
    
        todo!();
        /*
        : boolean(boolean),
        : ty(TensorIndexType::Boolean),

        
        */
    }
}

impl From<Slice> for TensorIndex {

    /// Case 5: Slice represented in `Slice` form
    fn from(slice: Slice) -> Self {
    
        todo!();
        /*
        : slice(move(slice)),
        : ty(TensorIndexType::Slice),

        
        */
    }
}

impl From<Tensor> for TensorIndex {

    /// Case 6: Tensor value
    fn from(tensor: Tensor) -> Self {
    
        todo!();
        /*
        : tensor(move(tensor)),
        : ty(TensorIndexType::Tensor),

        
        */
    }
}

impl TensorIndex {
    
    #[inline] pub fn is_none(&self) -> bool {
        
        todo!();
        /*
            return type_ == TensorIndexType::None;
        */
    }
    
    #[inline] pub fn is_ellipsis(&self) -> bool {
        
        todo!();
        /*
            return type_ == TensorIndexType::Ellipsis;
        */
    }
    
    #[inline] pub fn is_integer(&self) -> bool {
        
        todo!();
        /*
            return type_ == TensorIndexType::Integer;
        */
    }
    
    #[inline] pub fn integer(&self) -> i64 {
        
        todo!();
        /*
            return integer_;
        */
    }
    
    #[inline] pub fn is_boolean(&self) -> bool {
        
        todo!();
        /*
            return type_ == TensorIndexType::Boolean;
        */
    }
    
    #[inline] pub fn boolean(&self) -> bool {
        
        todo!();
        /*
            return boolean_;
        */
    }
    
    #[inline] pub fn is_slice(&self) -> bool {
        
        todo!();
        /*
            return type_ == TensorIndexType::Slice;
        */
    }
    
    #[inline] pub fn slice(&self) -> &Slice {
        
        todo!();
        /*
            return slice_;
        */
    }
    
    #[inline] pub fn is_tensor(&self) -> bool {
        
        todo!();
        /*
            return type_ == TensorIndexType::Tensor;
        */
    }
    
    #[inline] pub fn tensor(&self) -> &Tensor {
        
        todo!();
        /*
            return tensor_;
        */
    }
}

#[inline] pub fn apply_slice(
    self_:                      &Tensor,
    dim:                        i64,
    start:                      i64,
    stop:                       i64,
    step:                       i64,
    disable_slice_optimization: bool,
    self_device:                &Device,
    self_sizes:                 &&[i32]) -> Tensor {

    todo!();
        /*
            // TODO: implement negative step
      TORCH_CHECK_VALUE(step > 0, "step must be greater than zero");

      // Skip this optimization if we are tracing, as the trace may be polymorphic
      // over the shape of the `self` tensor, and we still want to record
      // the slice.
      i64 length = (self_device == kCPU || self_device == kCUDA) ? self_sizes[dim] : self.size(dim);
      if (!disable_slice_optimization && start == 0 && stop == length && step == 1) {
        return self;
      }
      return self.slice(dim, start, stop, step);
        */
}

#[inline] pub fn apply_select(
        self_:       &Tensor,
        dim:         i64,
        index:       i64,
        real_dim:    i64,
        self_device: &Device,
        self_sizes:  &&[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK_INDEX(
        !(index == 0 && dim == 0 && self_sizes.size() == 0),
        "invalid index of a 0-dim tensor. ",
        "Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number");

      i64 size = self_sizes[dim];
      TORCH_CHECK_INDEX(
        index >= -size && index < size,
        "index ", index, " is out of bounds for dimension ", real_dim, " with size ", size);

      // if the index is negative, do not normalize it because that would fix the index
      // on the current tensor size in the tracer.
      // select also works on negative indices
      return self.select(dim, index);
        */
}

#[inline] pub fn bool_to_indexing_tensor_cpu_orcuda(
        self_: &Tensor,
        value: bool) -> Tensor {
    
    todo!();
        /*
            // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
      if (value) {
        return empty({1}, {}, self.options().dtype(kLong)).fill_(0.);
      } else {
        return empty({0}, {}, self.options().dtype(kLong));
      }
        */
}


#[inline] pub fn bool_to_indexing_tensor_non_native_device_type(
        self_: &Tensor,
        value: bool) -> Tensor {
    
    todo!();
        /*
            // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
      if (value) {
        return zeros({1}, {}, self.options().dtype(kLong));
      } else {
        return empty({0}, {}, self.options().dtype(kLong));
      }
        */
}


#[inline] pub fn bool_to_indexing_tensor(
        self_:       &Tensor,
        value:       bool,
        self_device: &Device) -> Tensor {
    
    todo!();
        /*
            if (self_device == kCPU || self_device == kCUDA) {
        return boolToIndexingTensorCPUOrCUDA(self, value);
      } else {
        return boolToIndexingTensorNonNativeDeviceType(self, value);
      }
        */
}


#[inline] pub fn scalar_to_tensor_non_native_device_type(
        v:       &Scalar,
        options: &TensorOptions) -> Tensor {
    
    todo!();
        /*
            return scalar_tensor(v, options);
        */
}


#[inline] pub fn record_tensor_index(
        tensor:      &Tensor,
        out_indices: &mut Vec<Tensor>,
        dim_ptr:     *mut i64)  {
    
    todo!();
        /*
            // TODO: check scalarType
      outIndices.resize(*dim_ptr + 1);
      outIndices[*dim_ptr] = tensor;
      (*dim_ptr)++;
        */
}


#[inline] pub fn type_convert_indices(
        self_:   &Tensor,
        indices: Vec<Tensor>) -> LinkedList<Option<Tensor>> {
    
    todo!();
        /*
            List<optional<Tensor>> converted_inds;
      converted_inds.reserve(indices.size());
      for (const auto &i: indices){
        converted_inds.push_back(move(i));
      }
      return converted_inds;
        */
}

/**
  | NOTE: Why do we mirror instead of replace the
  | `count_specified_dimensions` function in
  | torch/csrc/autograd/python_variable_indexing.cpp?
  |
  | It's because `count_specified_dimensions` is on
  | the hot path of Python tensor multi-dim
  | indexing (i.e. it's called by `applySlicing`
  | which is called by `THPVariable_getitem`
  | / `THPVariable_setitem` when handling indexing
  | of more than one dimension).
  |
  | If we were to merge the Python/C++
  | `count_specified_dimensions` function, on the
  | Python side we would have to construct
  | a `vector` container to be consumed by the C++
  | `count_specified_dimensions` function, which
  | adds 100s of nanoseconds overhead and is
  | undesirable.
  |
  */
#[inline] pub fn count_specified_dimensions(indices: &&[TensorIndex]) -> i64 {
    
    todo!();
        /*
            // Count the number of indexed dimensions (everything but ellipsis and None)
      i64 count = 0;
      for (auto& obj : indices) {
        if (obj.is_tensor()) {
          auto& tensor = obj.tensor();
          if (tensor.scalar_type() == kByte || tensor.scalar_type() == kBool) {
            count += tensor.dim();
          } else {
            count++;
          }
        } else if (!obj.is_none() && !obj.is_ellipsis() && !obj.is_boolean()) {
          count++;
        }
      }
      return count;
        */
}

/**
  | NOTE: Many functions below are only for
  | consumption from Python indexing
  | implementation, they include:
  |
  | - `Tensor scalarToTensor(...)`
  | - `IntArrayRef slicePrefix1sSize(...)`
  | - `void copy_to(...)`
  | - `Tensor handleDimInMultiDimIndexing(...)`
  | - `Tensor dispatch_index(...)`
  | - `Tensor dispatch_index_put_(...)`
  | - `Tensor get_item(...)`
  | - `void set_item(...)`
  |
  | The rest of the functions are in
  | `impl` namespace, signifying that
  | they shouldn't be used from Python indexing
  | implementation.
  |
  */
#[inline] pub fn scalar_to_tensor(
        v:           &Scalar,
        options:     &TensorOptions,
        self_device: &Device) -> Tensor {
    
    todo!();
        /*
            if (self_device == kCPU) {
        return scalar_tensor_static(v, options.dtype_opt()->toScalarType(), self_device);
      } else {
        return scalarToTensorNonNativeDeviceType(v, options);
      }
        */
}

/**
  | To match numpy semantics:
  |
  | As a special case for backwards compatibility,
  | strip away unit dimensions from the left of
  | 'src'
  |
  */
#[inline] pub fn slice_prefix1s_size<'a>(sizes: &'a &'a [i32]) -> &'a [i32] {
    
    todo!();
        /*
            usize first_non1_src = sizes.size();
      for (usize i = 0; i < sizes.size(); ++i) {
        if (sizes[i] != 1) {
          first_non1_src = i;
          break;
        }
      }

      return sizes.slice(first_non1_src);
        */
}

#[inline] pub fn copy_to(
        dst: &Tensor,
        src: &Tensor)  {
    
    todo!();
        /*
            if (dst.sizes().equals(src.sizes())) {
        // A shortcut to avoid generating hard-coded constant sizes during tracing.
        // This is not a perfect solution: when src & dst have different shapes, constants will still
        // appear. Users can workaround that case by dst[index..] = src.reshape(..)
        dst.copy_(src);
        return;
      }
      auto src_view = src.view(slicePrefix1sSize(src.sizes()));
      MaybeOwned<Tensor> b_src = expand_inplace(dst, src_view, "setitem");
      dst.copy_(*b_src);
        */
}

/**
  | See NOTE [ Setting `disable_slice_optimization`
  | when calling C++ tensor indexing functions from
  | Python ]
  |
  */
#[inline] pub fn handle_dim_in_multi_dim_indexing(
        prev_dim_result:            &Tensor,
        original_tensor:            &Tensor,
        index:                      &TensorIndex,
        dim_ptr:                    *mut i64,
        specified_dims_ptr:         *mut i64,
        real_dim:                   i64,
        out_indices:                &mut Vec<Tensor>,
        disable_slice_optimization: bool,
        original_tensor_device:     &Device,
        prev_dim_result_sizes:      &&[i32]) -> Tensor {
    
    todo!();
        /*
            if (index.is_integer()) {
        return applySelect(prev_dim_result, *dim_ptr, index.integer(), real_dim, original_tensor_device, prev_dim_result_sizes);
      } else if (index.is_slice()) {
        Tensor result = applySlice(
          prev_dim_result,
          *dim_ptr,
          index.slice().start(),
          index.slice().stop(),
          index.slice().step(),
          /*disable_slice_optimization=*/disable_slice_optimization,
          original_tensor_device,
          prev_dim_result_sizes);
        (*dim_ptr)++;
        return result;
      } else if (index.is_ellipsis()) {
        (*dim_ptr) += original_tensor.dim() - (*specified_dims_ptr);
        return prev_dim_result;
      } else if (index.is_none()) {
        Tensor result = prev_dim_result.unsqueeze(*dim_ptr);
        (*dim_ptr)++;
        return result;
      } else if (index.is_boolean()) {
        Tensor result = prev_dim_result.unsqueeze(*dim_ptr);
        recordTensorIndex(boolToIndexingTensor(result, index.boolean(), original_tensor_device), outIndices, dim_ptr);
        return result;
      } else if (index.is_tensor()) {
        Tensor result = prev_dim_result;
        const Tensor& tensor = index.tensor();
        auto scalar_type = tensor.scalar_type();
        if (tensor.dim() == 0 && isIntegralType(scalar_type, /*includeBool=*/true)) {
          if (scalar_type != kByte && scalar_type != kBool) {
            result = applySelect(result, *dim_ptr, tensor.item<i64>(), real_dim, original_tensor_device, prev_dim_result_sizes);
          } else {
            result = result.unsqueeze(*dim_ptr);
            if (scalar_type == kBool) {
              recordTensorIndex(boolToIndexingTensor(result, tensor.item<bool>() != 0, original_tensor_device), outIndices, dim_ptr);
            } else {
              recordTensorIndex(boolToIndexingTensor(result, tensor.item<u8>() != 0, original_tensor_device), outIndices, dim_ptr);
            }
          }
        } else {
          recordTensorIndex(tensor, outIndices, dim_ptr);
        }
        return result;
      } else {
        TORCH_INTERNAL_ASSERT(false, "Invalid TensorIndex type");
      }
        */
}

/**
  | This mirrors `applySlicing` in torch/csrc/autograd/python_variable_indexing.cpp
  |
  */
#[inline] pub fn apply_slicing(
        self_:                      &Tensor,
        indices:                    &&[TensorIndex],
        out_indices:                &mut Vec<Tensor>,
        disable_slice_optimization: bool,
        self_device:                &Device,
        self_sizes:                 &&[i32]) -> Tensor {
    
    todo!();
        /*
            i64 dim = 0;
      i64 specified_dims = count_specified_dimensions(indices);

      TORCH_CHECK_INDEX(
        specified_dims <= (i64)self_sizes.size(),
        "too many indices for tensor of dimension ", (int)self_sizes.size());

      Tensor result = self;
      for (usize i = 0; i < indices.size(); i++) {
        auto& obj = indices[i];
        result = handleDimInMultiDimIndexing(
          /*prev_dim_result=*/result,
          /*original_tensor=*/self,
          /*index=*/obj,
          /*dim=*/&dim,
          /*specified_dims=*/&specified_dims,
          /*real_dim=*/i,
          /*outIndices=*/outIndices,
          /*disable_slice_optimization=*/disable_slice_optimization,
          /*original_tensor_device=*/self_device,
          /*prev_dim_result_sizes=*/result.sizes());
      }
      return result;
        */
}

#[inline] pub fn dispatch_index(
        self_:   &Tensor,
        indices: Vec<Tensor>) -> Tensor {
    
    todo!();
        /*
            return self.index(typeConvertIndices(self, move(indices)));
        */
}

#[inline] pub fn dispatch_index_put(
        self_:   &mut Tensor,
        indices: Vec<Tensor>,
        value:   &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.index_put_(typeConvertIndices(self, move(indices)), value);
        */
}

/**
  | NOTE [ Setting `disable_slice_optimization`
  | when calling C++ tensor indexing functions from
  | Python ]
  |
  | Question: When should we set
  | `disable_slice_optimization` to `true` when
  | calling C++ tensor indexing functions from
  | Python indexing code?
  |
  | Answer: What "slice optimization" means: when
  | we have a slicing expression like `x[0:5, 0]`,
  | where the sliced tensor was of size 5 in
  | dimension 0, we would skip dispatching the
  | actual slice call as an optimization. However,
  | here are the cases where we DON'T want this
  | optimization:
  |
  | 1. When we are doing 1-D slicing
  | (e.g. `tensor[:]`).
  |
  |    Reason: we always return a shallow copy for
  |    expressions such as `tensor[:]`
  |    / `tensor[...]` / `tensor[:, :]`.
  |
  |    (Note that for `tensor[:, :]`, we return an
  |    alias of `tensor` by doing the following:
  |    ```
  |    Tensor sliced = applySlicing(self, indices, tensorIndices, disable_slice_optimization, self_device, self_sizes);
  |    if (tensorIndices.empty()) {
  |      if (sliced.is_same(self)) {
  |        // ensure we return a shallow copy for things like x[...]
  |        sliced = alias(sliced);
  |      }
  |      return sliced;
  |    }
  |    ```)
  |
  | 2. When we are doing JIT tracing.
  |
  |    Reason: JIT tracing needs the
  |    `self.slice(...)` call to properly trace the
  |    slice operation.
  |
  | This mirrors `THPVariable_getitem` in
  | torch/csrc/autograd/python_variable_indexing.cpp
  |
  | See NOTE [ Setting `disable_slice_optimization`
  | when calling C++ tensor indexing functions from
  | Python ]
  |
  */
#[inline] pub fn get_item(
    self_:                      &Tensor,
    indices:                    &&[TensorIndex],
    disable_slice_optimization: Option<bool>) -> Tensor {

    let disable_slice_optimization: bool = disable_slice_optimization.unwrap_or(false);

    todo!();
        /*
            Device self_device = self.device();
      IntArrayRef self_sizes = self.sizes();

      // handle simple types: integers, slices, none, ellipsis, bool
      if (indices.size() == 1) {
        const TensorIndex& index = indices[0];
        if (index.is_integer()) {
          return applySelect(self, 0, index.integer(), 0, self_device, self_sizes);
        } else if (index.is_slice()) {
          return applySlice(
            self,
            0,
            index.slice().start(),
            index.slice().stop(),
            index.slice().step(),
            /*disable_slice_optimization=*/true,
            self_device,
            self_sizes);
        } else if (index.is_none()) {
          return self.unsqueeze(0);
        } else if (index.is_ellipsis()) {
          return alias(self);
        } else if (index.is_boolean()) {
          Tensor result = self.unsqueeze(0);
          return dispatch_index(
            result,
            vector<Tensor>{boolToIndexingTensor(result, index.boolean(), self_device)}
          );
        }
      }

      vector<Tensor> tensorIndices;
      Tensor sliced = applySlicing(self, indices, tensorIndices, disable_slice_optimization, self_device, self_sizes);
      if (tensorIndices.empty()) {
        if (sliced.is_same(self)) {
          // ensure we return a shallow copy for things like x[...]
          sliced = alias(sliced);
        }
        return sliced;
      }

      // indexing by tensors ("advanced" indexing)
      return dispatch_index(sliced, move(tensorIndices));
        */
}

/**
  | This mirrors `THPVariable_setitem` in
  | torch/csrc/autograd/python_variable_indexing.cpp
  | for "the assigned value is a Tensor" case
  |
  | See NOTE [ Setting `disable_slice_optimization`
  | when calling C++ tensor indexing functions from
  | Python ]
  |
  */
#[inline] pub fn set_item_with_tensor(
    self_:                      &Tensor,
    indices:                    &&[TensorIndex],
    value:                      &Tensor,
    disable_slice_optimization: Option<bool>)  {

    let disable_slice_optimization: bool = disable_slice_optimization.unwrap_or(false);

    todo!();
        /*
            Device self_device = self.device();
      IntArrayRef self_sizes = self.sizes();

      // handle simple types: integers, slices, ellipsis, bool
      if (indices.size() == 1) {
        const TensorIndex& index = indices[0];
        if (index.is_boolean() && !index.boolean()) {
          // do nothing for false (technically we should check the size, but we don't have
          // real 0-sized shapes.
          return;
        } else if (index.is_ellipsis()) {
          copy_to(self, value);
          return;
        } else if (index.is_none() || (index.is_boolean() && index.boolean())) {
          copy_to(self.unsqueeze(0), value);
          return;
        } else if (index.is_integer()) {
          copy_to(applySelect(self, 0, index.integer(), 0, self_device, self_sizes), value);
          return;
        } else if (index.is_slice()) {
          copy_to(applySlice(
            self,
            0,
            index.slice().start(),
            index.slice().stop(),
            index.slice().step(),
            /*disable_slice_optimization=*/disable_slice_optimization,
            self_device,
            self_sizes), value);
          return;
        }
      }

      vector<Tensor> tensorIndices;
      Tensor sliced = applySlicing(self, indices, tensorIndices, disable_slice_optimization, self_device, self_sizes);
      if (tensorIndices.empty()) {
        copy_to(sliced, value);
        return;
      }

      IntArrayRef valueSizes = value.sizes();
      IntArrayRef slicedValueSizes = slicePrefix1sSize(valueSizes);
      Tensor valuesSliced;
      if (!valueSizes.equals(slicedValueSizes)) {
        valuesSliced = value.view(slicedValueSizes);
      } else {
        valuesSliced = value;
      }
      dispatch_index_put_(sliced, move(tensorIndices), valuesSliced);
      return;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorIndexing.cpp]

pub const ELLIPSIS: EllipsisIndexType = EllipsisIndexType {};

impl fmt::Display for Slice {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << slice.start() << ":" << slice.stop() << ":" << slice.step();
      return stream;
        */
    }
}


impl fmt::Display for TensorIndex {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            if (tensor_index.is_none()) {
        stream << "None";
      } else if (tensor_index.is_ellipsis()) {
        stream << "...";
      } else if (tensor_index.is_integer()) {
        stream << tensor_index.integer();
      } else if (tensor_index.is_boolean()) {
        stream << boolalpha << tensor_index.boolean();
      } else if (tensor_index.is_slice()) {
        stream << tensor_index.slice();
      } else if (tensor_index.is_tensor()) {
        stream << tensor_index.tensor();
      }
      return stream;
        */
    }
}

/**
  | This mirrors `THPVariable_setitem` in
  | torch/csrc/autograd/python_variable_indexing.cpp
  | for "the assigned value is a Scalar" case
  |
  */
#[inline] pub fn set_item_with_scalar(
        self_:   &Tensor,
        indices: &[TensorIndex],
        v:       &Scalar)  {
    
    todo!();
        /*
            Tensor value;

      {
        AutoDispatchBelowADInplaceOrView guard;
        // TODO: This qint special case looks very suspicious...
        if (isQIntType(self.scalar_type())) {
          value = scalarToTensor(v, device(kCPU).dtype(kFloat), Device(kCPU));
        } else {
          value = scalarToTensor(v, self.options(), self.device());
        }
      }

      return set_item(self, indices, value);
        */
}

impl Tensor {
    
    pub fn index(&self, indices: &[TensorIndex]) -> Tensor {
        
        todo!();
        /*
            TORCH_CHECK(indices.size() > 0, "Passing an empty index list to Tensor::index() is not valid syntax");
      OptionalDeviceGuard device_guard(device_of(*this));
      return get_item(*this, indices);
        */
    }
    
    pub fn index_put_tensor(&mut self, 
        indices: &[TensorIndex],
        rhs:     &Tensor) -> &mut Tensor {
        
        todo!();
        /*
            TORCH_CHECK(indices.size() > 0, "Passing an empty index list to Tensor::index_put_() is not valid syntax");
      OptionalDeviceGuard device_guard(device_of(*this));
      set_item(*this, indices, rhs);
      return *this;
        */
    }
    
    pub fn index_put_scalar(&mut self, 
        indices: &[TensorIndex],
        v:       &Scalar) -> &mut Tensor {
        
        todo!();
        /*
            TORCH_CHECK(indices.size() > 0, "Passing an empty index list to Tensor::index_put_() is not valid syntax");
      OptionalDeviceGuard device_guard(device_of(*this));
      set_item(*this, indices, v);
      return *this;
        */
    }
}
