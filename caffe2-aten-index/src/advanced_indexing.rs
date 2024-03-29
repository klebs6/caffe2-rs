/**
  | “The very first thing I remember in my
  | early childhood is a flame, a blue flame
  | jumping off a gas stove somebody lit...
  | 
  | I remember being shocked by the whoosh
  | of the blue flame jumping off the burner,
  | the suddenness of it... I saw that flame
  | and felt that hotness of it close to my
  | face.
  | 
  | I felt fear, real fear, for the first
  | time in my life.
  | 
  | But I remember it also like some kind
  | of adventure, some kind of weird joy,
  | too. I guess that experience took me
  | someplace in my head I hadn't been before...
  | 
  | The fear I had was almost like an invitation,
  | a challenge to go forward into something
  | I knew nothing about.
  | 
  | That's where I think my personal philosophy
  | of life and my commitment to everything
  | I believe in started...
  | 
  | In my mind I have always believed and
  | thought since then that my motion had
  | to be forward, away from the heat of that
  | flame.”
  | 
  | ― Miles Davis
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorAdvancedIndexing.h]

/// Indexing tensors by by tensors
#[repr(u8)]
pub enum ScatterGatherOp {
    REDUCE_ADD, 
    REDUCE_MULTIPLY
}

pub type IndexFn = fn(
        _0:              &mut TensorIterator,
        indexed_sizes:   &[i32],
        indexed_strides: &[i32]
) -> ();

pub type IndexFillFn = fn(
        iter:            &mut TensorIterator,
        dim:             i64,
        self_dim_size:   i64,
        self_dim_stride: i64,
        source:          &Scalar
) -> ();

pub type IndexCopyFn = fn(
        iter:            &mut TensorIterator,
        dim:             i64,
        self_dim_size:   i64,
        self_dim_stride: i64
) -> ();

pub type IndexPutFn = fn(
        _0:              &mut TensorIterator,
        indexed_sizes:   &[i32],
        indexed_strides: &[i32],
        accumulate:      bool
) -> ();

pub type IndexPutWithSortFn = fn(
        _0:         &mut Tensor,
        _1:         &[Option<Tensor>],
        _2:         &Tensor,
        accumulate: bool,
        _unsafe:     bool
) -> ();

pub type MaskedFillFn = fn(_0: &mut TensorIterator, scalar: &Scalar) -> ();

pub type PutFn = fn(
        iter:       &mut TensorIterator,
        self_:      &Tensor,
        accumulate: bool
) -> ();

pub type TakeFn = fn(iter: &mut TensorIterator, input: &Tensor) -> ();

pub type MaskedSelectFn = fn(_0: &mut TensorIterator, orig_stride: i64) -> ();

pub type MaskedScatterFn = fn(_0: &mut TensorIterator, _1: &Tensor) -> ();

pub type GatherFn = fn(
        result: &mut Tensor,
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor
) -> ();

pub type ScatterFn = fn(
        self_: &mut Tensor,
        dim:   i64,
        index: &Tensor,
        src:   &Tensor
) -> ();

pub type ScatterFillFn = fn(
        self_: &mut Tensor,
        dim:   i64,
        index: &Tensor,
        src:   &Scalar
) -> ();

pub type ScatterAddFn = fn(
        self_: &mut Tensor,
        dim:   i64,
        index: &Tensor,
        src:   &Tensor
) -> ();

pub type ScatterReduceFn = fn(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        src:    &Tensor,
        reduce: &ScatterGatherOp
) -> ();

pub type ScatterScalarReduceFn = fn(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        value:  &Scalar,
        reduce: &ScatterGatherOp
) -> ();

declare_dispatch!{index_fn, index_stub}
declare_dispatch!{index_fill_fn, index_fill_stub}
declare_dispatch!{index_copy_fn, index_copy_stub}
declare_dispatch!{index_put_fn, index_put_stub}
declare_dispatch!{index_put_with_sort_fn, index_put_with_sort_stub}
declare_dispatch!{put_fn, put_stub}
declare_dispatch!{take_fn, take_stub}
declare_dispatch!{masked_fill_fn, masked_fill_stub}
declare_dispatch!{masked_select_fn, masked_select_serial_stub}
declare_dispatch!{masked_select_fn, masked_select_stub}
declare_dispatch!{masked_scatter_fn, masked_scatter_stub}

declare_dispatch!{gather_fn, gather_stub}
declare_dispatch!{scatter_fn, scatter_stub}
declare_dispatch!{scatter_fill_fn, scatter_fill_stub}
declare_dispatch!{scatter_add_fn, scatter_add_stub}
declare_dispatch!{scatter_reduce_fn, scatter_reduce_stub}
declare_dispatch!{scatter_scalar_reduce_fn, scatter_scalar_reduce_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorAdvancedIndexing.cpp]
// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in
// NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a &[Tensor] containg kLong, kBool
// or kByte tensors or nulls. 
//
// Byte tensors (boolean masks) are expanded to
// long tensors via nonzero(). 
//
// Null tensors signify that the dimension is not
// indexed.
//
// All indexes are broadcast together and iterated
// as *one*. From NumPy:
//
// result[i_1, ..., i_M] 
// == x[
//      ind_1[i_1, ..., i_M], 
//      ind_2[i_1, ..., i_M], 
//      ..., 
//      ind_N[i_1, ..., i_M]
//      ]
//
// Note 1: ByteTensors expand to index as many
// dimensions as there are in the mask.
//
// Note 2: The behavior is more complicated when
// the index tensors are not all adjacent
// (e.g. x[[0, 1], :, [2, 3]]). 
//
// In this case, self and the index tensors are
// transposed to the front: x.transpose(1, 2)[[0,
// 1], [2, 3]]
//
// The code contains two implementations of
// indexing. The more efficient implementation
// treats indexing like an elementwise operation
// over the tensors `result`, `x`, `ind_1`,
// `ind_2`, etc. 
//
// This implementation does not work for
// index_put_ with accumulate=True. The other
// implementation combines the indexed tensors
// into a single linear index that is used with
// Tensor.put_. This is used for index_put_ with
// accumulate=True.
//
// The more efficient implementation takes the
// following steps for the above operation:
//
// 1) Broadcast ind_1, ind_2, ind_3 together to
// a common shape
//
// 2) Record x.stride(i) for each indexed
// dimension `i`
//
// 3) Replace the indexed subspace of `x` with the
// shape of the corresponding subspace of
// `result` but with stride 0
//
// 4) Add dimensions of size 1 to the index
// tensors (ind_1, ind_2, etc.) so that their
// shape is compatible with the result shape
//
// The CPU or CUDA kernel then computes
// element-wise over the broadcasted and restrided
// result, x, ind_1,  ind_2, etc.:
//
//   result[...] = *(&x[...] +
//                   ind_1[...] * x.stride(1) +
//                   ind_2[...] * x.stride(2) +
//                   ...)
//
// where & and * represent the C-style address-of
// and indirection operations.
//
pub fn get_operator_enum(reduce: &str) -> ScatterGatherOp {
    
    todo!();
        /*
            if (reduce == "add") {
        return SCATTER_GATHER_OP::REDUCE_ADD;
      } else if (reduce == "multiply") {
        return SCATTER_GATHER_OP::REDUCE_MULTIPLY;
      } else {
        TORCH_CHECK(false, "reduce argument must be either add or multiply.");
      }
        */
}

pub fn scatter_meta_impl<Meta>(
        meta:   &mut Meta,
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor,
        src:    &Option<Tensor>,
        reduce: Option<&str>)  {

    todo!();
        /*
            i64 wrapped_dim = maybe_wrap_dim(dim, self.dim());
      scatter_gather_dtype_check("scatter", self, index, src);
      scatter_shape_check(self, wrapped_dim, index, src);
      auto output = meta.maybe_get_output(0);

      if (output.defined()) {
        assert_no_internal_overlap(output);
        assert_no_overlap(output, index);
        if (src.has_value()) {
          assert_no_overlap(output, src.value());
        }
      }

      meta.set_output(self.sizes(), self.options());
      if (reduce.has_value()) {
        // Check if we have a valid reduce operator.
        get_operator_enum(reduce.value());
      }
        */
}

lazy_static!{
    /*
    TORCH_META_FUNC2(scatter, src)
    (const Tensor& self, i64 dim, const Tensor& index, const Tensor& src) {
      scatter_meta_impl(*this, self, dim, index, src);
    }

    TORCH_META_FUNC2(scatter, value)
    (const Tensor& self, i64 dim, const Tensor& index, const Scalar& value) {
      scatter_meta_impl(*this, self, dim, index);
    }

    TORCH_META_FUNC2(scatter, reduce)
    (const Tensor& self,
     i64 dim,
     const Tensor& index,
     const Tensor& src,
     const string_view reduce) {
      scatter_meta_impl(*this, self, dim, index, src, reduce);
    }

    TORCH_META_FUNC2(scatter, value_reduce)
    (const Tensor& self,
     i64 dim,
     const Tensor& index,
     const Scalar& src,
     const string_view reduce) {
      scatter_meta_impl(*this, self, dim, index, nullopt, reduce);
    }

    TORCH_META_FUNC(scatter_add)
    (const Tensor& self, i64 dim, const Tensor& index, const Tensor& src) {
      scatter_meta_impl(*this, self, dim, index, src, "add");
    }
    */
}

define_dispatch!{index_stub}
define_dispatch!{index_fill_stub}
define_dispatch!{index_copy_stub}
define_dispatch!{index_put_stub}
define_dispatch!{index_put_with_sort_stub}
define_dispatch!{put_stub}
define_dispatch!{take_stub}
define_dispatch!{masked_fill_stub}
define_dispatch!{masked_select_serial_stub}
define_dispatch!{masked_select_stub}
define_dispatch!{masked_scatter_stub}
define_dispatch!{gather_stub}
define_dispatch!{scatter_stub}
define_dispatch!{scatter_fill_stub}
define_dispatch!{scatter_add_stub}
define_dispatch!{scatter_reduce_stub}
define_dispatch!{scatter_scalar_reduce_stub}

register_no_cpu_dispatch!{index_put_with_sort_stub, index_put_with_sort_fn}

pub fn all_strides_match(tensors: &[Tensor]) -> bool {
    
    todo!();
        /*
            TORCH_CHECK(tensors.size() >= 1);
      auto strides = tensors[0].strides();
      for (auto& tensor : tensors.slice(1)) {
        if (!strides.equals(tensor.strides())) {
          return false;
        }
      }
      return true;
        */
}

pub fn shapes_as_str(tensors: &[Tensor]) -> String {
    
    todo!();
        /*
            ostringstream os;
      bool first = true;
      for (auto& tensor : tensors) {
        if (tensor.defined()) {
          if (!first) {
            os << ", ";
          }
          os << tensor.sizes();
          first = false;
        }
      }
      return os.str();
        */
}

/**
  | Replace indexed dimensions in src with stride
  | 0 and the size of the result tensor.
  |
  | The offset in these dimensions is computed by
  | the kernel using the index tensor's values and
  | the stride of src. The new shape is not
  | meaningful. It's used to make the shape
  | compatible with the result tensor.
  |
  */
pub fn restride_src(
        src:               &Tensor,
        dims_before:       i64,
        dims_indexed:      i64,
        replacement_shape: &[i32]) -> Tensor {
    
    todo!();
        /*
            auto shape = DimVector(src.sizes());
      auto strides = DimVector(src.strides());
      i64 end = dims_before + dims_indexed;
      shape.erase(shape.begin() + dims_before, shape.begin() + end);
      strides.erase(strides.begin() + dims_before, strides.begin() + end);
      shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
      strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
      return src.as_strided(shape, strides);
        */
}

/**
  | Add dimensions of size 1 to an index tensor so
  | that it can be broadcast to the result shape
  | and iterated over element-wise like the result
  | tensor and the restrided src.
  |
  */

pub fn reshape_indexer(
        index:       &Tensor,
        dims_before: i64,
        dims_after:  i64) -> Tensor {
    
    todo!();
        /*
            auto orig_shape = index.sizes();
      auto shape = DimVector();
      shape.append(dims_before, 1);
      shape.append(orig_shape.begin(), orig_shape.end());
      shape.append(dims_after, 1);
      return index.reshape(shape);
        */
}

//-------------------------------------
pub struct AdvancedIndex {
    src:             Tensor,
    indices:         Vec<Tensor>,
    indexed_sizes:   DimVector,
    indexed_strides: DimVector,
    dims_before:     i64,
    dims_after:      i64,
}

impl AdvancedIndex {
    
    pub fn new(
        src:          &Tensor,
        indices_list: &[Tensor]) -> Self {
    
        todo!();
        /*


            i64 element_size_bytes = src.element_size();
      i64 dims_before = 0, dims_after = 0, dims_indexed = 0;
      IntArrayRef replacement_shape;
      for (usize dim = 0; dim < indices_list.size(); dim++) {
        if (!indices_list[dim].defined()) {
          if (dims_indexed == 0) {
            dims_before++;
          } else {
            dims_after++;
          }
        } else {
          dims_indexed++;
          replacement_shape = indices_list[dim].sizes();
          indexed_sizes.push_back(src.size(dim));
          indexed_strides.push_back(src.stride(dim) * element_size_bytes);
        }
      }

      // Check if the indexed subspace contains a dim of size 0, but the replacement
      // shape does not. This implies that an index is out of bounds, because there
      // is no number that's a valid index for an empty tensor. Normally, out of
      // bounds is handled in the indexing kernel, but this case fails earlier in
      // restride_src with an unhelpful error message.
      if (find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
          find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
        TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
      }

      this->dims_before = dims_before;
      this->dims_after = dims_after;
      this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

      for (auto& index : indices_list) {
        if (index.defined()) {
          indices.push_back(reshape_indexer(index, dims_before, dims_after));
        }
      }

      // For CUDA tensors, force all index tensors to have the same striding to
      // simplify the CUDA kernel.
      if (indices.size() >= 2 && this->src.device().type() == kCUDA) {
        if (!all_strides_match(indices)) {
          for (usize i = 0; i < indices.size(); i++) {
            indices[i] = indices[i].contiguous();
          }
        }
      }
        */
    }
}



pub fn make_info(
    self_: Tensor,
    orig:  &[Option<Tensor>]

) -> AdvancedIndex {
    
    todo!();
        /*
            checkIndexTensorTypes(orig);
      // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
      auto indices = expandTensors(self, orig);
      // next broadcast all index tensors together
      try {
        indices = expand_outplace(indices);
      } catch (exception& e) {
        TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                       " with shapes ", shapes_as_str(indices));
      }
      // add missing null Tensors so that it matches self.dim()
      while (indices.size() < (usize)self.dim()) {
        indices.emplace_back();
      }
      // if the non-null indices are not all adjacent, transpose self and indices
      // together so that they're adjacent at the front
      if (!hasContiguousSubspace(indices)) {
        tie(self, indices) = transposeToFront(self, indices);
      }
      // Ensure indices are on the same device as self
      for (usize i = 0; i < indices.size(); i++) {
        if (indices[i].defined() && indices[i].device() != self.device()) {
          indices[i] = indices[i].to(self.device());
        }
      }
      return AdvancedIndex(self, indices);
        */
}

pub fn make_index_put_iterator<'a>(
    info:  &AdvancedIndex,
    value: &Tensor

) -> TensorIterator<'a> {
    
    todo!();
        /*
            TORCH_CHECK(is_expandable_to(value.sizes(), info.src.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
                 " cannot be broadcast to indexing result of shape ", info.src.sizes());
      TORCH_CHECK(value.scalar_type() == info.src.scalar_type(),
                  "Index put requires the source and destination dtypes match, "
                  "got ", info.src.scalar_type(), " for the destination "
                  "and ", value.scalar_type(), " for the source.");
      TensorIteratorConfig config;
      // info.src is restrided by restride_src with 0 strided dimensions
      config.set_check_mem_overlap(false);
      config.resize_outputs(false);
      config.check_all_same_dtype(false);
      config.add_output(info.src);
      config.add_input(value);
      for (auto& index : info.indices) {
        config.add_input(index);
      }
      return config.build();
        */
}

pub fn make_index_iterator(info: &AdvancedIndex) -> TensorIterator {
    
    todo!();
        /*
            TensorIteratorConfig config;
      config.set_check_mem_overlap(false)
            .check_all_same_dtype(false)
            .declare_static_dtype_and_device(info.src.scalar_type(), info.src.device())
            .add_owned_output(Tensor())
            .add_input(info.src);
      for (auto& index : info.indices) {
        config.add_input(index);
      }
      return config.build();
        */
}


pub fn make_index_out_iterator<'a>(
    info:   &AdvancedIndex,
    result: &mut Tensor

) -> TensorIterator<'a> {
    
    todo!();
        /*
            TensorIteratorConfig config;
      // info.src is a restrided view of result
      config.set_check_mem_overlap(false)
            .check_all_same_dtype(false)
            .add_output(result)
            .add_input(info.src);
      for (auto& index : info.indices) {
        config.add_input(index);
      }
      return config.build();
        */
}


pub fn index(
    self_:   &Tensor,
    indices: &[Option<Tensor>]

) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK_INDEX(indices.size() <= (usize)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");

      auto info = make_info(self, indices);
      auto iter = make_index_iterator(info);
      index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
      return iter.output();
        */
}


pub fn quantized_index(
    self_:   &Tensor,
    indices: &[Option<Tensor>]

) -> Tensor {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(
          self.qscheme() == kPerTensorAffine ||
          self.qscheme() == kPerTensorSymmetric,
          "Indexing is only supported for per-Tensor quantized Tensors.");

      // For now, this is a naive implementation which does dq -> index -> q.
      // TODO(future PR): improve performance by removing the copies.
      const auto& self_dq = self.dequantize();

      TORCH_CHECK_INDEX(indices.size() <= (usize)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");

      auto info = make_info(self_dq, indices);
      auto iter = make_index_iterator(info);
      index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
      Tensor res = iter.output();

      return quantize_per_tensor(
          res, self.q_scale(), self.q_zero_point(), self.scalar_type());
        */
}

pub fn index_out<'a>(
    result:  &mut Tensor,
    self_:   &Tensor,
    indices: &[Option<Tensor>]

) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK_INDEX(indices.size() <= (usize)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
      assert_no_internal_overlap(result);
      assert_no_overlap(result, self);
      for (const optional<Tensor>& index: indices) {
        if (index.has_value()) {
          assert_no_overlap(result, *index);
        }
      }

      auto info = make_info(self, indices);
      auto iter = make_index_out_iterator(info, result);
      index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
      return result;
        */
}

pub fn put_a<'a>(
    self_:      &mut Tensor,
    index:      &Tensor,
    source:     &Tensor,
    accumulate: bool

) -> &'a mut Tensor {
    
    todo!();
        /*
            // See note [Writing Nondeterministic Operations]
      // Nondeterministic when index contains duplicate entries and we do not accumulate
      // If we accumulate on GPU, we use atomicGPUAdd, which is non-deterministic
      if (!accumulate || (accumulate && self.device().type() == DeviceType::Cuda)) {
        globalContext().alertNotDeterministic("put_");
      }

      // Type and device checks
      TORCH_CHECK(index.scalar_type() == ScalarType::Long, "put_(): Expected a long tensor for index, but got ", index.scalar_type())
      TORCH_CHECK(self.scalar_type() == source.scalar_type(), "put_(): self and source expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and source.dtype = ", source.scalar_type());
      TORCH_CHECK(self.device() == source.device() && self.device() == index.device(),
          "put_(): self, index and source expected to be in the same device, but got self.device = ",
          self.device(), ", index.device = ", index.device(), ", and source.device = ", source.device());

      // index checks
      TORCH_CHECK_INDEX(source.numel() == index.numel(), "put_(): Expected source and index to have the same number of elements, but got source.numel() = ", source.numel(), ", index.numel() = ", index.numel());
      TORCH_CHECK_INDEX(!(self.numel() == 0 && index.numel() != 0), "put_(): Tried to put elements into an empty tensor");

      assert_no_internal_overlap(self);
      assert_no_overlap(self, index);
      assert_no_overlap(self, source);

      // Early return
      if (index.numel() == 0) {
        return self;
      }

      auto index_reshaped = index.reshape(source.sizes());
      // Do not iterate over self, we will compute the offsets manually
      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_input(source)
        .add_input(index_reshaped)
        .build();

      put_stub(iter.device_type(), iter, self, accumulate);

      return self;
        */
}

pub fn put_b(
    self_:      &Tensor,
    index:      &Tensor,
    source:     &Tensor,
    accumulate: bool) -> Tensor {
    
    todo!();
        /*
            return self.clone(MemoryFormat::Preserve).put_(index, source, accumulate);
        */
}


pub fn index_put_a(
        self_:      &Tensor,
        indices:    &[Option<Tensor>],
        value:      &Tensor,
        accumulate: bool) -> Tensor {
    
    todo!();
        /*
            return self.clone(MemoryFormat::Preserve).index_put_(indices, value, accumulate);
        */
}


pub fn index_put_impl<'a>(
        self_:      &mut Tensor,
        indices:    &[Option<Tensor>],
        value:      &Tensor,
        accumulate: bool,
        unsafe_:    bool) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK_INDEX(indices.size() <= (usize)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
      if (has_internal_overlap(self) == MemOverlap::YES) {
        TORCH_WARN(
          "Use of index_put_ on expanded tensors is deprecated. "
          "Please clone() the tensor before performing this operation. "
          "This also applies to advanced indexing e.g. tensor[indices] = tensor");
      }
      assert_no_overlap(self, value);
      for (const optional<Tensor>& index: indices) {
        if (index.has_value()) {
          assert_no_overlap(self, *index);
        }
      }

      if (self.device().type() == DeviceType::Cuda && (accumulate || globalContext().deterministicAlgorithms())) {
          TORCH_CHECK(value.device() == self.device(), "expected device ", self.device(), " but got device ",
          value.device(), " for value tensor");
          index_put_with_sort_stub(self.device().type(), self, indices, value, accumulate, unsafe);
          return self;
      }

      auto info = make_info(self, indices);
      auto iter = make_index_put_iterator(info, value);
      index_put_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides, accumulate);
      return self;
        */
}


pub fn take_out<'a>(
        self_: &Tensor,
        index: &Tensor,
        out:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // Type and device checks
      TORCH_CHECK(index.scalar_type() == ScalarType::Long, "take(): Expected a long tensor for index, but got ", index.scalar_type())
      TORCH_CHECK(self.scalar_type() == out.scalar_type(), "take(): self and out expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and out.dtype = ", out.scalar_type());
      TORCH_CHECK(self.device() == out.device() && self.device() == index.device(),
          "take(): self, index and out expected to be in the same device, but got self.device = ",
          self.device(), ", index.device = ", index.device(), ", and out.device = ", out.device());

      // index checks
      TORCH_CHECK_INDEX(!(self.numel() == 0 && index.numel() != 0), "take(): tried to take from an empty tensor");

      assert_no_internal_overlap(out);
      assert_no_overlap(out, index);
      assert_no_overlap(out, self);

      // Do not iterate over self, we will compute the offsets manually
      // out is resized inside tensor_iterator
      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_output(out)
        .add_input(index)
        .build();

      // Early return after out has been resized
      if (index.numel() == 0) {
        return out;
      }

      take_stub(iter.device_type(), iter, self);

      return out;
        */
}

pub fn take(
        self_: &Tensor,
        index: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto out = empty(index.sizes(), self.options());
        take_out(self, index, out);
        return out;
        */
}

pub fn index_put_b<'a>(
    self_:      &mut Tensor,
    indices:    &[Option<Tensor>],
    value:      &Tensor,
    accumulate: bool) -> &'a mut Tensor {
    
    todo!();
        /*
            return _index_put_impl_(self, indices, value, accumulate, /*unsafe=*/false);
        */
}


pub fn index_copy_a<'a>(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            dim = maybe_wrap_dim(dim, self.dim());

      TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");
      assert_no_internal_overlap(self);
      assert_no_overlap(self, index);
      assert_no_overlap(self, source);

      i64 numIndices = index.numel();
      if (source.dim() == 0 && numIndices != 1) {
        TORCH_CHECK_INDEX(false, "index_copy_(): When source is scalar, index should have one element (got ", numIndices, ")");
      } else if ((source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
        TORCH_CHECK_INDEX(false, "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
                       source.dim(), "), destination dimensionality (", self.dim(), ")");
      }

      TORCH_CHECK(index.scalar_type() == ScalarType::Long, "index_copy_(): Expected a long tensor for index, but got ", index.scalar_type())
      TORCH_CHECK(self.scalar_type() == source.scalar_type(), "index_copy_(): self and source expected to have the same dtype, but got (self) ", self.scalar_type(), " and (source) ", source.scalar_type());
      TORCH_CHECK(self.device() == source.device() && self.device() == index.device(),
          "index_copy_(): self, index and source expected to be in the same device, but got (self) ",
          self.device(), ", (index) ", index.device(), ", and (source) ", source.device());

      // Check that source and destination slices have the same size
      auto selfSlicedSizes = self.sizes().vec();
      if (selfSlicedSizes.size() > 0) {
        selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
      }
      auto sourceSlicedSizes = source.sizes().vec();
      if (sourceSlicedSizes.size() > 0) {
        sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
      }
      if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
          !equal(selfSlicedSizes.begin(), selfSlicedSizes.end(),
                      sourceSlicedSizes.begin())) {
        stringstream ss;
        ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
        ss << "Destination slice shape: " << selfSlicedSizes << " at dimension " << dim;
        ss << " and source slice shape: " << sourceSlicedSizes << " at dimension 0.";
        TORCH_CHECK(false, ss.str());
      }
      TORCH_CHECK_INDEX(source.dim() == 0 || numIndices == source.size(dim),
              "index_copy_(): Number of indices (", numIndices, ") should be equal to source.size(dim) (", source.size(dim), ")");

      // See Note [Enabling Deterministic Operations]
      if (self.device().type() == DeviceType::Cuda && globalContext().deterministicAlgorithms()){
        TorchList<optional<Tensor>> indices;
        indices.reserve(dim + 1);
        for (const auto i: irange(dim)) {
          (void)i;
          indices.emplace_back();
        }
        indices.emplace_back(index);
        return self.index_put_(indices, source, false);
      }

      return _index_copy_(self, dim, index, source);
        */
}


pub fn index_copy_impl<'a>(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // Handle the case when self / source is 0-dim
      Tensor self_nonzero = self.dim() == 0 ? self.unsqueeze(0) : self;
      Tensor source_nonzero = source.dim() == 0 ? source.unsqueeze(0) : source;

      // The only different between the following  tensor iterator and that of index_fill_ is that
      // this one has also source as an input. We should refactor it when if constexpr is available (C++17)

      // Prepare `index` for TensorIterator.
      // It is restrided to be broadcastable over `self` in TensorIterator.
      auto index_sizes = vector<i64>(self_nonzero.dim(), 1);
      auto index_strides = vector<i64>(self_nonzero.dim(), 0);
      index_sizes[dim] = index.numel();
      index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
      auto index_restrided = index.as_strided(
        index_sizes, index_strides);

      // Prepare `self` for TensorIterator.
      // Restride `self` to not advance in dimension `dim`.
      // We do not use squash_dim here because `index` will
      // need to advance in this dimension.
      // Note that self_sizes[dim] is set to index.numel().
      // This is done so that self_sizes[dim] and index_sizes[dim]
      // match as required by TensorIterator (input shape should
      // strictly broadcast over output shape, i.e.
      // output.shape[i] >= input.shape[i] for i in range(dims)).
      auto self_sizes = self_nonzero.sizes().vec();
      auto self_strides = self_nonzero.strides().vec();
      self_sizes[dim] = index.numel();
      self_strides[dim] = 0;
      auto self_restrided = self_nonzero.as_strided(self_sizes, self_strides);

      auto iter = TensorIteratorConfig()
        // We do not check for overlap because `self` is restrided
        // with zero stride. Zero strides trigger memory overlap assert
        // within TensorIterator.
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(self_restrided)
        .add_input(index_restrided)
        .add_input(source_nonzero)
        .build();

      auto self_dim_size = self_nonzero.size(dim);
      auto self_dim_stride = self_nonzero.stride(dim);
      index_copy_stub(
        iter.device_type(),
        iter,
        dim,
        self_dim_size,
        self_dim_stride);

      return self;
        */
}

pub fn index_copy_b(
    self_:  &Tensor,
    dim:    i64,
    index:  &Tensor,
    source: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.clone(MemoryFormat::Preserve).index_copy_(dim, index, source);
        */
}


pub fn index_add_cpu<'a>(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Tensor,
        alpha:  &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            dim = maybe_wrap_dim(dim, self.dim());

      auto numel = index.numel();
      TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector");
      TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              "index_add_(): Expected dtype int32/int64 for index");
      TORCH_CHECK(self.scalar_type() == source.scalar_type(),
                  "index_add_(): self and source must have the same scalar type");
      TORCH_CHECK(dim == 0 || dim < source.dim(),
                  "index_add_(): Indexing dim ", dim, " is out of bounds of tensor");
      TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
                  "index_add_(): Number of indices should be equal to self.size(dim)");

      assert_no_internal_overlap(self);
      assert_no_overlap(self, index);
      assert_no_overlap(self, source);

      auto index_contig = index.contiguous();

      if (self.dim() > 1) {
        // Equivalent to:
        //   for (auto i = 0; i < numel; i++) {
        //     auto selfSlice = self.select(dim, index_data[i]);
        //     auto sourceSlice = source.select(dim, i);
        //     selfSlice.add_(sourceSlice);
        //   }
        // But much faster as this reuses the iterator from add_
        if (numel == 0) {
          return self;
        }
        auto selfSlice = self.select(dim, 0);
        auto sourceSlice = source.select(dim, 0);
        auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
        auto source_stride_bytes = source.stride(dim) * elementSize(source.scalar_type());
        auto self_dim_size = self.size(dim);
        auto iter = TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);

        AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cpu_", [&] () {
          auto index_data = index_contig.data_ptr<Index>();
          for (auto i = 0; i < numel; i++) {
              auto self_i = index_data[i];
              TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
              auto self_data = static_cast<char*>(selfSlice.data_ptr()) + self_i * self_stride_bytes;
              auto source_data = static_cast<char*>(sourceSlice.data_ptr()) + i * source_stride_bytes;
              iter.unsafe_replace_operand(0, self_data);
              iter.unsafe_replace_operand(1, self_data);
              iter.unsafe_replace_operand(2, source_data);
              add_stub(iter.device_type(), iter, alpha);
          }
        });
      }
      else {
        TORCH_CHECK(source.dim() <= 1, "source.dim() (", source.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");

        // explicitly capture all required variables to work around windows build
        // TODO: fix this when windows can correctly capture variables in nested lambda
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
          self.scalar_type(), "index_add_", [&self, &source, &dim, &index_contig, &numel, &alpha] {
          auto alpha_value = alpha.to<Scalar>();
          auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
          auto source_stride = source.dim() == 0 ? 1 : source.stride(dim);
          // TODO: Maybe TensorAccessor can beused here?
          auto* self_ptr = self.data_ptr<Scalar>();
          auto* source_ptr = source.data_ptr<Scalar>();
          AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_add_cpu_",
            [&index_contig, &numel, &self, &self_ptr, &self_stride, &source_ptr, &source_stride, alpha_value] {
            auto index_data = index_contig.data_ptr<Index>();
            for (auto i = 0; i < numel; i++) {
                auto self_i = index_data[i];
                TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self.numel()), "index out of range in self");
                Scalar *self_ip = self_ptr + self_i * self_stride;
                *self_ip += *(source_ptr + i * source_stride) * alpha_value;
            }
          });
        });
      }
      return self;
        */
}


pub fn index_add_a<'a>(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return self.index_add_(dim, index, source, 1);
        */
}


pub fn index_add_b(
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Tensor,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.clone(MemoryFormat::Preserve).index_add_(dim, index, source, alpha);
        */
}

pub fn index_add_c(
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.clone(MemoryFormat::Preserve).index_add_(dim, index, source);
        */
}

/**
  | Check that indices fall within dimension
  | array size
  | 
  | Avoid redispatch call to min/max
  |
  */
pub fn check_indexarray_range<IndexType>(
        indices:           *const IndexType,
        n:                 i64,
        indexing_axis_dim: IndexType)  {

    todo!();
        /*
            for (const auto i : irange(n)) {
        auto idx = indices[i];
        TORCH_CHECK(
            0 <= idx && idx < indexing_axis_dim,
            "INDICES element is out of DATA bounds, id=",
            idx,
            " axis_dim=",
            indexing_axis_dim);
      }
        */
}


pub fn index_select_out_cpu_dim1<'a>(
        result_contig: &mut Tensor,
        self_:         &Tensor,
        index_contig:  &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto self_contig = self.contiguous();
      const TypeMeta dataType = self_contig.dtype();
      usize item_bytesize = dataType.itemsize();

      auto out = static_cast<char*>(result_contig.data_ptr());

      auto src_base = static_cast<const char*>(self_contig.data_ptr());

      auto self_sizes = self_contig.sizes();
      auto outer_dims_product = Sizeo_dim_(1, self_sizes);
      auto block_size = size_from_dim_(2, self_sizes);
      auto block_bytesize = block_size * item_bytesize;

      auto src_indexing_axis_dim = self_sizes[1];
      auto src_batch_bytesize = self_sizes[1] * block_bytesize;
      auto N = index_contig.numel();

      auto gathered_batch_bytesize = N * block_bytesize;

      AT_DISPATCH_INDEX_TYPES(
        index_contig.scalar_type(), "batch_index_select_compute", [&]() {

          const auto* idxs = index_contig.data_ptr<Index>();
          check_indexarray_range<Index>(idxs, N, src_indexing_axis_dim);

          // Special-case single-float copy for efficiency
          if (self.scalar_type() == ScalarType::Float && block_size == 1) {
            for (auto batch = 0; batch < outer_dims_product; ++batch) {
              const float* src_floats =
                  (const float*)(src_base + batch * src_batch_bytesize);
              float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);

              for (auto i = 0; i < N; ++i) {
                auto idx = idxs[i];
                if (idx < 0) {
                  idx = idx + src_indexing_axis_dim;
                }
                dst_floats[i] = src_floats[idx];
              }
            }
          } else {
            // outer_dims_product specifies how many times we repeat inner dimensions,
            // so we just iterate over it to cover all outer dimensions.
            for (auto batch = 0; batch < outer_dims_product; ++batch) {
              for (auto i = 0; i < N; ++i) {
                auto idx = idxs[i];
                if (idx < 0) {
                  idx = idx + src_indexing_axis_dim;
                }

                auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
                auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
                memcpy(dst, src, block_bytesize);
              }
            }
          }
      });
      return result_contig;
        */
}


pub fn index_select_out_cpu<'a>(
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            dim = maybe_wrap_dim(dim, self.dim());

      auto numel = index.numel();
      TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
      TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
      TORCH_CHECK(self.scalar_type() == result.scalar_type(),
                  "index_select(): self and result must have the same scalar type");
      TORCH_CHECK(dim == 0 || dim < self.dim(),
                  "index_select(): Indexing dim ", dim, " is out of bounds of tensor");
      assert_no_internal_overlap(result);
      assert_no_overlap(result, self);
      assert_no_overlap(result, index);

      auto result_size = self.sizes().vec();
      if (self.dim() > 0) {
        result_size[dim] = numel;
      }
      resize_output(result, result_size);

      auto index_contig = index.contiguous();

      if (self.dim() > 1) {
        if (numel == 0 || self.numel() == 0) {
          return result;
        }

        if (dim == 1 && result.is_contiguous()) {
          // fast pass
          return index_select_out_cpu_dim1_(result, self, index_contig);
        }

        auto selfSlice = self.select(dim, 0);
        auto resultSlice = result.select(dim, 0);
        auto selfSlice_data = selfSlice.data_ptr();
        auto resultSlice_data = resultSlice.data_ptr();
        auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
        auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
        auto self_dim_size = self.size(dim);
        auto slice_size = selfSlice.numel();

        auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .resize_outputs(false)
          .add_output(resultSlice)
          .add_input(selfSlice)
          .build();

        auto grain_size = internal::GRAIN_SIZE;
        auto outer_loop =
          // explicitly capture all required variables to work around windows build
          // TODO: fix this when windows can correctly capture variables in nested lambda
          [&index_contig, &iter, &self_dim_size, &selfSlice_data, &self_stride_bytes, &resultSlice_data,
            &result_stride_bytes](i64 start, i64 end) {
          auto sub_iter = TensorIterator(iter);
          AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
            [&index_contig, &start, &end, &sub_iter, &self_dim_size, &selfSlice_data, &self_stride_bytes,
              &resultSlice_data, &result_stride_bytes] () {
            auto index_data = index_contig.data_ptr<Index>();
            for (i64 i = start; i < end; i++) {
              auto self_i = index_data[i];
              TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
              auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
              auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
              sub_iter.unsafe_replace_operand(0, result_data);
              sub_iter.unsafe_replace_operand(1, self_data);
              copy_stub(sub_iter.device_type(), sub_iter, false);
            };
          });
        };

        // parallel on inner loop in case the slice is large enough;
        // otherwise parallel on outer loop
        if (slice_size >= grain_size) {
          outer_loop(0, numel);
        } else {
          // use a fast loop when self and result are contiguous and of the same data type
          if (iter.is_contiguous() && self.scalar_type() == result.scalar_type()) {
            auto slice_size_bytes = slice_size * elementSize(self.scalar_type());
            // explicitly capture all required variables to work around windows build
            // TODO: fix this when windows can correctly capture variables in nested lambda
            parallel_for(0, numel, grain_size / slice_size,
              [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
                &self_stride_bytes, &resultSlice_data, &result_stride_bytes](i64 start, i64 end) {
              AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
                [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
                  &self_stride_bytes, &resultSlice_data, &result_stride_bytes, &start, &end] () {
                auto index_data = index_contig.data_ptr<Index>();
                for (i64 i = start; i < end; i++) {
                  auto self_i = index_data[i];
                  TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
                  auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
                  auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
                  memcpy(result_data, self_data, slice_size_bytes);
                }
              });
            });
          } else {
            parallel_for(0, numel, grain_size / slice_size, outer_loop);
          }
        }
      } else {
        TORCH_CHECK(result.dim() <= 1, "result.dim() (", result.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");
        // explicitly capture all required variables to work around windows build
        // TODO: fix this when windows can correctly capture variables in nested lambda
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
          self.scalar_type(), "index_select", [&index_contig, &self, &result, &dim, &numel] {
          auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
          auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);

          auto self_data_ptr = self.data_ptr<Scalar>();
          auto result_data_ptr = result.data_ptr<Scalar>();
          auto self_numel = self.numel();
          AT_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
            [&index_contig, &numel, &self_numel, &self_data_ptr, &self_stride, &result_data_ptr, &result_stride] {
            auto index_data = index_contig.data_ptr<Index>();
            for (auto i = 0; i < numel; i++) {
              auto self_i = index_data[i];
              TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_numel), "index out of range in self");
              Scalar *self_ip = self_data_ptr + self_i * self_stride;
              *(result_data_ptr + i * result_stride) = *self_ip;
            }
          });
        });
      }

      return result;
        */
}


pub fn index_select_cpu(
        self_: &Tensor,
        dim:   i64,
        index: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return index_select_out_cpu_(self, dim, index, result);
        */
}


pub fn index_select_backward(
        grad:       &Tensor,
        self_sizes: &[i32],
        dim:        i64,
        index:      &Tensor) -> Tensor {
    
    todo!();
        /*
            return zeros(self_sizes, grad.options()).index_add_(dim, index, grad);
        */
}

pub fn index_fill_a<'a>(
        self_:  &mut Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;

      TORCH_CHECK_INDEX(
        index.scalar_type() == ScalarType::Long,
        "index_fill_(): Expected dtype int64 for index.");

      assert_no_overlap(self, index);
      if (has_internal_overlap(self) == MemOverlap::YES) {
        TORCH_WARN(
          "Use of index_fill_ on expanded tensors is deprecated. "
          "Please clone() the tensor before performing this operation. "
          "This also applies to advanced indexing e.g. tensor[mask] = scalar");
      }

      if (!self.is_complex() && source.isComplex()) {
        TORCH_CHECK(false, "index_fill_(): Converting complex Scalar to non-complex type is not supported");
      }

      // Handle the case when `self` is 0-dim
      Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;

      dim = maybe_wrap_dim(dim, self_nonzero_dim);
      TORCH_CHECK(index.dim() <= 1, "Index has to be a vector/scalar");

      // Prepare `index` for TensorIterator.
      // It is restrided to be broadcastable over `self` in TensorIterator.
      auto index_sizes = vector<i64>(self_nonzero_dim.dim(), 1);
      auto index_strides = vector<i64>(self_nonzero_dim.dim(), 0);
      index_sizes[dim] = index.numel();
      index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
      auto index_restrided = index.as_strided(
        index_sizes, index_strides);

      // Prepare `self` for TensorIterator.
      // Restride `self` to not advance in dimension `dim`.
      // We do not use squash_dim here because `index` will
      // need to advance in this dimension.
      // Note that self_sizes[dim] is set to index.numel().
      // This is done so that self_sizes[dim] and index_sizes[dim]
      // match as required by TensorIterator (input shape should
      // strictly broadcast over output shape, i.e.
      // output.shape[i] >= input.shape[i] for i in range(dims)).
      auto self_sizes = self_nonzero_dim.sizes().vec();
      auto self_strides = self_nonzero_dim.strides().vec();
      self_sizes[dim] = index.numel();
      self_strides[dim] = 0;
      auto self_restrided = self_nonzero_dim.as_strided(self_sizes, self_strides);

      auto iter = TensorIteratorConfig()
        // We do not check for overlap because `self` is restrided
        // with zero stride. Zero strides trigger memory overlap assert
        // within TensorIterator.
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(self_restrided)
        .add_input(index_restrided)
        .build();

      auto self_dim_size = (self_nonzero_dim.sizes())[dim];
      auto self_dim_stride = (self_nonzero_dim.strides())[dim];
      index_fill_stub(
        iter.device_type(),
        iter,
        dim,
        self_dim_size,
        self_dim_stride,
        source);

      return self;
        */
}

pub fn index_fill_b<'a>(
    self_:  &mut Tensor,
    dim:    i64,
    index:  &Tensor,
    source: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(source.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
          "with ", source.dim(), " dimension(s).");
      return self.index_fill_(dim, index, source.item());
        */
}


pub fn index_fill_c(
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Scalar) -> Tensor {
    
    todo!();
        /*
            return self.clone(MemoryFormat::Preserve).index_fill_(dim, index, source);
        */
}


pub fn index_fill_d(
        self_:  &Tensor,
        dim:    i64,
        index:  &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            return self.clone(MemoryFormat::Preserve).index_fill_(dim, index, source);
        */
}


pub fn gather_out_cpu_cuda<'a>(
        self_:       &Tensor,
        dim:         i64,
        index:       &Tensor,
        sparse_grad: bool,
        result:      &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            resize_output(result, index.sizes());
      assert_no_internal_overlap(result);
      assert_no_overlap(result, self);
      assert_no_partial_overlap(result, index);
      gather_stub(result.device().type(), result, self, dim, index);
      return result;
        */
}


pub fn gather(
        self_:       &Tensor,
        dim:         i64,
        index:       &Tensor,
        sparse_grad: bool) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return gather_out_cpu_cuda(self, dim, index, sparse_grad, result);
        */
}


pub fn gather_backward(
        grad:        &Tensor,
        self_:       &Tensor,
        dim:         i64,
        index:       &Tensor,
        sparse_grad: bool) -> Tensor {
    
    todo!();
        /*
            if (sparse_grad) {
        return _gather_sparse_backward(self, dim, index, grad);
      }
      return zeros(self.sizes(), grad.options()).scatter_add_(dim, index, grad);
        */
}



pub fn scatter_impl<T, ReduceStub, FillStub>(
    self_:       &Tensor,
    dim:         i64,
    index:       &Tensor,
    src:         &T,
    out:         &Tensor,
    reduce_stub: &mut ReduceStub,
    fill_stub:   &mut FillStub,
    reduce:      Option<&str>

) {

    todo!();
        /*
            auto mut_out = const_cast<Tensor&>(out);

      if (!self.is_same(mut_out)) {
        mut_out.copy_(self);
      }

      if (reduce.has_value()) {
        auto op = meta::get_operator_enum(reduce.value());
        reduce_stub(self.device().type(), mut_out, dim, index, src, op);
      } else {
        fill_stub(self.device().type(), mut_out, dim, index, src);
      }
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(scatter_src_out)
    (const Tensor& self,
     i64 dim,
     const Tensor& index,
     const Tensor& src,
     const Tensor& out) {
      scatter_impl(self, dim, index, src, out,
                   scatter_reduce_stub,
                   scatter_stub);
    }

    TORCH_IMPL_FUNC(scatter_value_out)
    (const Tensor& self,
     i64 dim,
     const Tensor& index,
     const Scalar& value,
     const Tensor& out) {
      scatter_impl(self, dim, index, value, out,
                   scatter_scalar_reduce_stub,
                   scatter_fill_stub);
    }

    TORCH_IMPL_FUNC(scatter_reduce_out)
    (const Tensor& self,
     i64 dim,
     const Tensor& index,
     const Tensor& src,
     const string_view reduce,
     const Tensor& out) {
      scatter_impl(self, dim, index, src, out,
                   scatter_reduce_stub,
                   scatter_stub,
                   reduce);
    }

    TORCH_IMPL_FUNC(scatter_value_reduce_out)
    (const Tensor& self,
     i64 dim,
     const Tensor& index,
     const Scalar& value,
     const string_view reduce,
     const Tensor& out) {
      scatter_impl(self, dim, index, value, out,
                   scatter_scalar_reduce_stub,
                   scatter_fill_stub,
                   reduce);
    }

    TORCH_IMPL_FUNC(scatter_add)
    (const Tensor& self,
     i64 dim,
     const Tensor& index,
     const Tensor& src,
     const Tensor& out) {
      auto mut_out = const_cast<Tensor&>(out);

      if (!self.is_same(mut_out)) {
        mut_out.copy_(self);
      }

      if (globalContext().deterministicAlgorithms() && self.device().type() == DeviceType::Cuda && self.dim() == 1) {
        TORCH_CHECK(index.dim() == 1 && src.dim() == 1, "index and src should be 1D tensors when self is a 1D tensor, "
          "but their dims are ", index.dim(), " and ", src.dim(), ", respectively");
        TORCH_CHECK(index.numel() == src.numel(), "index and src should have same number of elements for 1D tensors, "
          "but got ", index.numel(), " versus ", src.numel());
        TORCH_CHECK(dim == 0, "dim should be zero for 1D self tensor, but got ", dim);
        TorchList<optional<Tensor>> indices;
        indices.reserve(1);
        indices.push_back(index);
        mut_out.index_put_(indices, src, true);
      } else {
        scatter_add_stub(self.device().type(), mut_out, dim, index, src);
      }
    }
    */
}


pub fn masked_scatter(
        self_:  &Tensor,
        mask:   &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            MaybeOwned<Tensor> _mask, _self;
      tie(_mask, _self) = expand_outplace(mask, self);
      return _self->clone(MemoryFormat::Contiguous).masked_scatter_(*_mask, source);
        */
}


pub fn masked_fill_impl_cpu<'a>(
        self_: &mut Tensor,
        mask:  &Tensor,
        value: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;
      if (mask.dtype() == ScalarType::Byte) {
        TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
                "please use a mask with dtype torch.bool instead.");
      }

      if (has_internal_overlap(self) == MemOverlap::YES) {
        TORCH_WARN(
          "Use of masked_fill_ on expanded tensors is deprecated. "
          "Please clone() the tensor before performing this operation. "
          "This also applies to advanced indexing e.g. tensor[mask] = scalar");
      }
      assert_no_partial_overlap(self, mask);

      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // deprecated, but not a hard error
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(self)
        .add_input(mask)
        .build();

      masked_fill_stub(iter.device_type(), iter, value);
      return self;
        */
}


pub fn masked_fill_cpu_a<'a>(
    self_: &mut Tensor,
    mask:  &Tensor,
    value: &Scalar) -> &'a mut Tensor {
    
    todo!();
        /*
            auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

      masked_fill_impl_cpu(self, mask, value);
      namedinference::propagate_names_if_nonempty(self, maybe_outnames);
      return self;
        */
}

pub fn masked_fill_cpu_b<'a>(
    self_: &mut Tensor,
    mask:  &Tensor,
    value: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
      TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
          "with ", value.dim(), " dimension(s).");

      masked_fill_impl_cpu(self, mask, value.item());
      namedinference::propagate_names_if_nonempty(self, maybe_outnames);
      return self;
        */
}


pub fn masked_fill_a(
        self_:  &Tensor,
        mask:   &Tensor,
        source: &Scalar) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
      {
        NoNamesGuard guard;
        MaybeOwned<Tensor> _mask, _self;
        tie(_mask, _self) = expand_outplace(mask, self);
        result = _self->clone(MemoryFormat::Contiguous);
        result.masked_fill_(mask, source);
      }
      namedinference::propagate_names_if_nonempty(result, maybe_outnames);
      return result;
        */
}

pub fn masked_fill_b(
        self_:  &Tensor,
        mask:   &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result;
      auto maybe_outnames = namedinference::broadcast_to_outnames(mask, self, "masked_fill");
      {
        NoNamesGuard guard;
        MaybeOwned<Tensor> _mask, _self;
        tie(_mask, _self) = expand_outplace(mask, self);
        result = _self->clone(MemoryFormat::Contiguous);
        result.masked_fill_(mask, source);
      }
      namedinference::propagate_names_if_nonempty(result, maybe_outnames);
      return result;
        */
}


pub fn masked_select_out_impl_cpu<'a>(
        result: &mut Tensor,
        self_:  &Tensor,
        mask:   &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;

      TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
                  "masked_select: expected BoolTensor or ByteTensor for mask");
      TORCH_CHECK(self.scalar_type() == result.scalar_type(),
                  "masked_select(): self and result must have the same scalar type");

      assert_no_internal_overlap(result);
      assert_no_overlap(result, self);
      assert_no_overlap(result, mask);

      if (mask.dtype() == ScalarType::Byte) {
        TORCH_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
                "please use a mask with dtype torch.bool instead.");
      }

      MaybeOwned<Tensor> _mask, _self;
      tie(_mask, _self) = expand_outplace(mask, self);

      auto shape = _self->sizes();
      i64 numel = _mask->sum().item().toLong();
      resize_output(result, {numel});
      if (numel == 0) {
        return result;
      }

      // Create strided view of result before feeding into TensorIterator
      auto strides = DimVector(shape.size(), 0);
      auto orig_stride = result.strides()[0];
      auto result_strided = result.as_strided(shape, strides);

      // serial kernel
      // serial kernel requires that src is traversed in its logical order. However, TensorIterator might
      // have reordered dimensions so that src would be traversed in its physical order, producing wrong
      // answers. A sufficient condition that no reorder happened is that both _self and _mask is contiguous.
      // If it is not satisfied, use parallel kernel that handles permutations correctly
      bool use_serial_kernel = (self.numel() < internal::GRAIN_SIZE || get_num_threads() == 1 ) &&
      _self->is_contiguous() && _mask->is_contiguous();
      if (use_serial_kernel) {
        auto iter = TensorIteratorConfig()
          .set_check_mem_overlap(false)  // result is intenionally zero-strided above
          .check_all_same_dtype(false)
          .resize_outputs(false)
          .add_output(result_strided)
          .add_input(*_self)
          .add_input(*_mask)
          .build();

        masked_select_serial_stub(iter.device_type(), iter, orig_stride);
        return result;
      }

      // Use a prefix sum to record the output locations of the masked elements,
      // so as to parallel with TensorIterator.
      auto mask_long = empty(shape, self.options().dtype(kLong)).copy_(*_mask);
      auto mask_prefix_sum = empty(shape, self.options().dtype(kLong));
      auto mask_long_data = mask_long.data_ptr<i64>();
      auto mask_prefix_sum_data = mask_prefix_sum.data_ptr<i64>();
      // TODO: Here can only use partial_sum for C++14,
      // use exclusive_scan when PyTorch upgrades to C++17, which have better peformance.
      // exclusive_scan(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data, 0);
      partial_sum(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data);

      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // result is intenionally zero-strided above
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(result_strided)
        .add_input(*_self)
        .add_input(*_mask)
        .add_input(mask_prefix_sum)
        .build();

      masked_select_stub(iter.device_type(), iter, orig_stride);
      return result;
        */
}


pub fn masked_select_out_cpu<'a>(
        self_:  &Tensor,
        mask:   &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            namedinference::compute_broadcast_outnames(self, mask);
      return masked_select_out_impl_cpu(result, self, mask);
        */
}


pub fn masked_select_cpu(
        self_: &Tensor,
        mask:  &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor result = empty({0}, self.options());
      return masked_select_out_cpu(self, mask, result);
        */
}


pub fn masked_select_backward(
        grad:  &Tensor,
        input: &Tensor,
        mask:  &Tensor) -> Tensor {
    
    todo!();
        /*
            // The following could just be written as `zeros_like(input).masked_scatter(mask, grad)`.
      // However, as an optimization, we call the in-place variant of masked_scatter.
      // Unfortunately, that doesn't allow for the broadcasting of the LHS, so we need
      // to explicitly broadcast here (the out-of-place variant of masked_scatter
      // implicitly handles broadcasting).
      auto result = zeros_like(
          input.expand(infer_size(input.sizes(), mask.sizes())), MemoryFormat::Preserve);
      return result.masked_scatter_(mask, grad);
        */
}

#[inline] pub fn take_along_dim_helper(
        self_:   &Tensor,
        indices: &Tensor,
        dim:     i64) -> (Tensor,Tensor,i64) {
    
    todo!();
        /*
            TORCH_CHECK(
          self.dim() == indices.dim(),
          "torch.take_along_dim(): input and indices should have the same number of dimensions, ",
          "but got ", self.dim(), " dimensions for input, and ", indices.dim(), " dimensions for indices")
      TORCH_CHECK(
          indices.scalar_type() == ScalarType::Long,
          "torch.take_along_dim(): dtype of indices should be Long but got ", indices.scalar_type())

      dim = maybe_wrap_dim(dim, self.dim());

      DimVector self_sizes{self.sizes()};
      // update number of elements at dim as per indices
      self_sizes[dim] = indices.size(dim);
      auto broadcast_shape = infer_size(self_sizes, indices.sizes());
      auto indices_broadcasted = broadcast_to(indices, broadcast_shape);

      DimVector indices_sizes{indices.sizes()};
      // update number of elements at dim as per self
      indices_sizes[dim] = self.size(dim);
      broadcast_shape = infer_size(indices_sizes, self.sizes());
      auto self_broadcasted = broadcast_to(self, broadcast_shape);

      return make_tuple(self_broadcasted, indices_broadcasted, dim);
        */
}

#[inline] pub fn check_device_a(
        c:      CheckedFrom,
        t:      &Tensor,
        device: Device)  {
    
    todo!();
        /*
            TORCH_CHECK(
          !t.defined() || t.device() == device,
          "Expected tensor to have ", device,
          " Device, but got tensor with ", t.device(), " Device ",
          "(while checking arguments for ", c, ")");
        */
}

#[inline] pub fn check_device_b(
        c:       CheckedFrom,
        tensors: &[Tensor],
        device:  Device)  {
    
    todo!();
        /*
            for (auto &t : tensors) {
        checkDevice(c, t, device);
      }
        */
}


pub fn take_along_dim(
        self_:   &Tensor,
        indices: &Tensor,
        opt_dim: Option<i64>) -> Tensor {
    
    todo!();
        /*
            checkDevice("torch.take_along_dim():", {self, indices}, self.device());
      if (opt_dim.has_value()) {
        i64 dim;
        Tensor self_broadcasted, indices_broadcasted;
        tie(self_broadcasted, indices_broadcasted, dim) =
            _take_along_dim_helper(self, indices, opt_dim.value());
        return self_broadcasted.gather(dim, indices_broadcasted);
      }

      // similar to `take`, but `take` doesn't support the same dtypes as `gather`.
      return self.view(-1).gather(0, indices.view(-1));
        */
}


pub fn take_along_dim_out<'a>(
        self_:   &Tensor,
        indices: &Tensor,
        opt_dim: Option<i64>,
        result:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            checkDevice("torch.take_along_dim():", {self, indices, result}, self.device());
      if (opt_dim.has_value()) {
        i64 dim;
        Tensor self_broadcasted, indices_broadcasted;
        tie(self_broadcasted, indices_broadcasted, dim) =
            _take_along_dim_helper(self, indices, opt_dim.value());
        return gather_out(result, self_broadcasted, dim, indices_broadcasted);
      }

      // similar to `take`, but `take` doesn't support the same dtypes as `gather`.
      return gather_out(result, self.view(-1), 0, indices.view(-1));
        */
}


pub fn gather_sparse_backward(
        self_: &Tensor,
        dim:   i64,
        index: &Tensor,
        grad:  &Tensor) -> Tensor {
    
    todo!();
        /*
            // special case scalar input and/or index
        if (self.ndimension() == 0) return _sparse_coo_tensor_unsafe(empty({0,grad.numel()}, index.options()), grad, self.sizes());
        if (grad.ndimension() == 0) return _sparse_coo_tensor_unsafe(index.view({1,1}), grad, self.sizes());
        Tensor sparse_ind = empty({self.ndimension(), grad.numel()}, self.options().dtype(kLong));
        i64 n_above = grad.numel();
        i64 n_below = 1;
        if (dim < 0) dim += self.ndimension();
        for (int i=0; i<self.ndimension(); i++) {
            n_above /= grad.size(i);
            if (i == dim) {
                sparse_ind[i] = index.reshape(-1);
            } else {
                sparse_ind[i] = arange(grad.size(i),self.options().dtype(kLong)).unsqueeze(1).expand({grad.size(i), n_above}).reshape(-1).repeat(n_below);
            }
            n_below *= grad.size(i);
        }
        return _sparse_coo_tensor_unsafe(sparse_ind, grad.reshape(-1), self.sizes());
        */
}



pub fn count_nonzero_impl<Scalar>(
    iter:  &mut TensorIteratorBase,
    range: Range<usize>

) -> i64 {

    todo!();
        /*
            i64 num_nonzero = 0;

      auto loop = [&](char** data, const i64* strides, i64 n) {
        constexpr int ilp_factor = 4;
        const char* ptr = data[0];
        const auto stride = strides[0];
        i64 nonzero[ilp_factor] = {0};

        i64 i = 0;
        for (; i + (ilp_factor - 1) < n; i += ilp_factor) {
          ForcedUnroll<ilp_factor>{}([&](int k) {
            const auto& val = *reinterpret_cast<const Scalar*>(ptr + k * stride);
            if (val != Scalar(0)) {
              ++nonzero[k];
            }
          });
          ptr += ilp_factor * stride;
        }
        for (; i < n; ++i) {
          const auto& val = *reinterpret_cast<const Scalar*>(ptr);
          if (val != Scalar(0)) {
            ++nonzero[0];
          }
          ptr += stride;
        }
        for (i64 k = 1; k < ilp_factor; ++k) {
          nonzero[0] += nonzero[k];
        }
        num_nonzero += nonzero[0];
      };
      iter.serial_for_each(loop, range);

      return num_nonzero;
        */
}


pub fn count_nonzero_cuda(
        self_: &Tensor,
        dims:  &[i32]) -> Tensor {
    
    todo!();
        /*
            return (self != 0).sum(dims);
        */
}


pub fn count_nonzero_cpu(
        self_: &Tensor,
        dims:  &[i32]) -> Tensor {
    
    todo!();
        /*
            if (dims.size() > 0) {
        return (self != 0).sum(dims);
      }

      // Optimized all-reduce
      auto iter = TensorIteratorConfig()
          .add_input(self)
          .build();

      const auto num_threads = get_num_threads();
      DimVector thread_count_nonzero(num_threads);

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_count_cpu", [&] {
        parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&] (i64 begin, i64 end) {
          const auto tid = get_thread_num();
          thread_count_nonzero[tid] = count_nonzero_impl<Scalar>(iter, {begin, end});
        });
      });

      for (i64 i = 1; i < num_threads; ++i) {
        thread_count_nonzero[0] += thread_count_nonzero[i];
      }
      auto out = empty({}, self.options().dtype(kLong));
      *out.data_ptr<i64>() = thread_count_nonzero[0];
      return out;
        */
}


pub fn count_nonzero(
        self_: &Tensor,
        dim:   Option<i64>) -> Tensor {
    
    todo!();
        /*
            if (dim) {
        return count_nonzero(self, IntArrayRef{*dim});
      }
      return count_nonzero(self, IntArrayRef{});
        */
}


pub fn nonzero_out_cpu<'a>(
        self_:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(result.scalar_type() == kLong,
                  "nonzero: Expected out tensor to have scalar type Long "
                  "but got scalar type", result.scalar_type());
      assert_no_internal_overlap(result);
      assert_no_overlap(result, self);

      auto iter = TensorIteratorConfig()
        .add_input(self)
        .enforce_linear_iteration()
        .build();

      const auto numel = iter.numel();
      const auto num_threads = get_num_threads();
      DimVector thread_begin(num_threads, -1);
      DimVector thread_count_nonzero(num_threads + 1);

      // Pass 1: Count nonzero element per-thread
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_count_cpu", [&] {
        parallel_for(0, numel, internal::GRAIN_SIZE, [&] (i64 begin, i64 end) {
          const auto tid = get_thread_num();
          thread_begin[tid] = begin;
          thread_count_nonzero[tid + 1] = count_nonzero_impl<Scalar>(iter, {begin, end});
        });
      });

      // Convert thread-local counts to cumulative sum
      for (usize i = 1; i < thread_count_nonzero.size(); ++i) {
        thread_count_nonzero[i] += thread_count_nonzero[i - 1];
      }

      const auto self_sizes = self.sizes();
      const auto total_nonzero = thread_count_nonzero.back();
      const i64 ndim = self_sizes.size();
      if (resize_output(result, {total_nonzero, ndim})) {
        // Default to fortran-contiguous output (see gh-46224)
        result.as_strided_({total_nonzero, ndim}, {1, total_nonzero});
      }

      if (result.numel() == 0) {
        return result;
      }

      // Pass 2: Write indexes
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          kHalf, kBFloat16, kBool, self.scalar_type(), "nonzero_cpu", [&] {
        parallel_for(0, numel, internal::GRAIN_SIZE, [&] (i64 begin, i64 end) {
          auto tid = get_thread_num();
          // Work needs to be distributed the same on both passes
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(begin == thread_begin[tid]);

          // +1 faster than additional condition check inside loop
          SmallVector<i64, 33> sizes(ndim + 1, -1);
          copy(self_sizes.begin(), self_sizes.end(), sizes.begin() + 1);
          SmallVector<i64, 33> current_idx(ndim + 1);
          if (begin > 0) {
            auto idx = begin;
            for (i64 k = ndim; idx > 0 && k > 0; --k) {
              current_idx[k] = idx % sizes[k];
              idx /= sizes[k];
            }
          }

          auto out_accessor = result.accessor<i64, 2>();
          auto out_ptr = out_accessor[thread_count_nonzero[tid]].data();

          auto loop = [&](char** data, const i64* strides, i64 n1, i64 n2) {
            // Copy into local variables to improve compiler alias analysis
            i64*  local_idx = current_idx.data() + 1;
            const i64*  local_sizes = sizes.data() + 1;
            const auto in_stride = strides[0];
            const auto out_stride1 = out_accessor.stride(1);
            const auto out_stride0 = out_accessor.stride(0) - ndim * out_stride1;
            const auto ndim = out_accessor.size(1);
            i64* out = out_ptr;

            for (i64 i = 0; i < n2; ++i) {
              const char* ptr = data[0] + i * strides[1];
              for (i64 j = 0; j < n1; ++j) {
                const auto& val = *reinterpret_cast<const Scalar*>(ptr);
                // If nonzero, write index
                if (val != Scalar(0)) {
                  for (i64 k = 0; k < ndim; ++k) {
                    *out = local_idx[k];
                    out += out_stride1;
                  }
                  out += out_stride0;
                }
                ptr += in_stride;

                // Advance current index
                i64 k = ndim - 1;
                ++local_idx[k];
                while (C10_UNLIKELY(local_idx[k] == local_sizes[k])) {
                  local_idx[k] = 0;
                  --k;
                  ++local_idx[k];
                }
              }
            }
            out_ptr = out;
          };
          iter.serial_for_each(loop, {begin, end});
          TORCH_INTERNAL_ASSERT(out_ptr == out_accessor[thread_count_nonzero[tid + 1]].data());
        });
      });
      return result;
        */
}


pub fn nonzero_cpu(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            auto result = empty({0}, self.options().dtype(kLong));
      nonzero_out_cpu(self, result);
      return result;
        */
}


pub fn nonzero_numpy(self_: &Tensor) -> Vec<Tensor> {
    
    todo!();
        /*
            // special case scalar for compatibility with numpy:
      //
      // >>> np.array(5).nonzero()
      // (array([0]),)
      // >>> np.array(0).nonzero()
      // (array([], dtype=int64),)

      if (self.dim() == 0) {
        return self.unsqueeze(0).nonzero().unbind(1);
      }

      return self.nonzero().unbind(1);
        */
}

pub fn masked_scatter_cpu<'a>(
        self_:  &mut Tensor,
        mask:   &Tensor,
        source: &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            assert_no_internal_overlap(self);
      TORCH_CHECK(
          self.scalar_type() == source.scalar_type(),
          "masked_scatter: expected self and source to have same dtypes but got",
          self.scalar_type(),
          " and ",
          source.scalar_type());

      TORCH_CHECK(self.device().type() == kCPU, "device type of self (", self.device().type(), ") is not CPU");
      TORCH_CHECK(mask.device().type() == kCPU, "device type of mask (", mask.device().type(), ") is not CPU");
      TORCH_CHECK(source.device().type() == kCPU, "device type of source (", source.device().type(), ") is not CPU");

      MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_scatter_");

      if (b_mask->dtype() == ScalarType::Byte) {
        TORCH_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
                "please use a mask with dtype torch.bool instead.");
      }

      auto src_cont = source.contiguous();

      auto iter = TensorIteratorConfig()
          .set_check_mem_overlap(false)
          .check_all_same_dtype(false)
          .resize_outputs(false)
          .add_output(self)
          .add_input(*b_mask)
          .build();

      masked_scatter_stub(iter.device_type(), iter, src_cont);
      return self;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/IndexingUtils.h]

pub fn invalid_mask(
    self_:    &Tensor,
    idx:      i64,
    mask:     &Tensor,
    mask_idx: i64
) {

    todo!();
        /*
            TORCH_CHECK_INDEX(false, "The shape of the mask ", mask.sizes(), " at index ", maskIdx,
      " does not match the shape of the indexed tensor ", self.sizes(), " at index ", idx);
        */
}

pub fn expand_tensors(
    self_:   &Tensor,
    indices: &[Option<Tensor>]

) -> Vec<Tensor> {

    todo!();
        /*
            // If indices come in as ByteTensor or BoolTensor (masks), expand them into the equivalent indexing by LongTensors
      vector<Tensor> result;
      for (optional<Tensor> index_opt : indices) {
        if (!index_opt.has_value()) {
          result.emplace_back();
        } else {
          Tensor index = move(*index_opt);
          if (index.scalar_type() == kByte || index.scalar_type() == kBool) {
            if (index.scalar_type() == kByte) {
              TORCH_WARN("indexing with dtype torch.uint8 is now deprecated," \
              " please use a dtype torch.bool instead.");
            }
            // The sizes of the ByteTensor mask or bool tensor must match the sizes of the
            // corresponding dimensions in self
            for (i64 j = 0; j < index.dim(); j++) {
              i64 srcIdx = result.size() + j;
              if (index.size(j) != self.size(srcIdx)) {
                invalid_mask(self, srcIdx, index, j);
              }
            }
            // Replace with nonzeros
            auto nonzero = index.nonzero();
            for (i64 j = 0; j < index.dim(); j++) {
              result.emplace_back(nonzero.select(1, j));
            }
          } else {
            result.emplace_back(move(index));
          }
        }
      }
      return result;
        */
}

pub fn check_index_tensor_types(indices: &[Option<Tensor>])  {
    
    todo!();
        /*
            for (optional<Tensor> tensor : indices) {
        if (tensor.has_value() && tensor->defined()) {
          auto scalarType = tensor->scalar_type();
          if (scalarType != kLong && scalarType != kByte && scalarType != kBool) {
              TORCH_CHECK_INDEX(false, "tensors used as indices must be long, byte or bool tensors");
          }
        }
      }
        */
}

#[inline] pub fn to_list_of_optional_tensors_a(list: &[Tensor]) 
-> &[Option<Tensor>] 
{
    todo!();
        /*
            TorchList<optional<Tensor>> result;
      result.reserve(list.size());
      for (const Tensor& a : list) {
        result.push_back(a);
      }
      return result;
        */
}

#[inline] pub fn to_list_of_optional_tensors_b(list: &[IValue]) 
-> &[Option<Tensor>] 
{
    
    todo!();
        /*
            TorchList<optional<Tensor>> result;
      result.reserve(list.size());
      for (const IValue& a : list) {
        result.push_back(a.toTensor());
      }
      return result;
        */
}

pub fn has_contiguous_subspace(tl: &[Tensor]) -> bool {
    
    todo!();
        /*
            // true if all the non-null tensors are adjacent
      auto isDefined = [](const Tensor & tensor){ return tensor.defined(); };
      auto isNull = [](const Tensor & tensor){ return !tensor.defined(); };
      auto start = find_if(tl.begin(), tl.end(), isDefined);
      auto stop = find_if(tl.rbegin(), tl.rend(), isDefined);
      auto it = find_if(start, stop.base(), isNull);
      return it == stop.base();
        */
}

/**
  | Transposes the tensor and indices together so
  | that all the non-null indices index the first
  | k dimensions of the tensor.
  |
  | Returns the transposed tensor and the reordered
  | indices.
  |
  | For example:
  |
  | transposeToFront(tensor, {nullptr, a, nullptr, b})
  |
  | returns
  |
  | tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
  */
pub fn transpose_to_front(
        self_:   Tensor,
        indices: &[Tensor]) -> (Tensor,Vec<Tensor>) {
    
    todo!();
        /*
            vector<i64> dims;
      vector<Tensor> transposedIndices;
      dims.reserve(self.dim());
      for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
        if (indices[i].defined()) {
          dims.push_back(i);
          transposedIndices.emplace_back(indices[i]);
        }
      }
      for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
        if (!indices[i].defined()) {
          dims.push_back(i);
          transposedIndices.emplace_back();
        }
      }
      return make_tuple(self.permute(dims), move(transposedIndices));
        */
}

#[inline] pub fn transpose_to_front_and_inv_perm(
        self_:   Tensor,
        indices: &[Tensor]) -> (Tensor,Vec<Tensor>,Vec<i64>) {
    
    todo!();
        /*
            vector<i64> dims;
      vector<i64> invPerm;
      vector<Tensor> transposedIndices;
      dims.reserve(self.dim());
      invPerm.resize(self.dim());
      for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
        if (indices[i].defined()) {
          dims.push_back(i);
          transposedIndices.emplace_back(indices[i]);
        }
      }
      for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
        if (!indices[i].defined()) {
          dims.push_back(i);
          transposedIndices.emplace_back();
        }
      }
      for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
        invPerm[dims[i]] = i;
      }
      return make_tuple(self.permute(dims), move(transposedIndices), move(invPerm));
        */
}


//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/IndexingUtils.cpp]

pub fn can_use_32bit_index_math(
        t:        &Tensor,
        max_elem: Option<i64>) -> bool {

    let max_elem: i64 = max_elem.unwrap_or(i32::max);
    
    todo!();
        /*
            i64 elements = t.numel();
      if (elements >= max_elem) {
        return false;
      }
      if (elements == 0) {
        return max_elem > 0;
      }

      i64 offset = 0;
      i64 linearId = elements - 1;

      // NOTE: Assumes all strides are positive, which is true for now
      for (int i = t.dim() - 1; i >= 0; --i) {
        i64 curDimIndex = linearId % t.size(i);
        i64 curDimOffset = curDimIndex * t.stride(i);
        offset += curDimOffset;
        linearId /= t.size(i);
      }

      if (offset >= max_elem) {
        return false;
      }

      return true;
        */
}
