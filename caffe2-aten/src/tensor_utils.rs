crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorUtils.h]

/**
  | The following are utility functions for
  | checking that arguments make sense.
  |
  | These are particularly useful for native
  | functions, which do NO argument checking by
  | default.
  */
pub struct TensorArg {
    tensor: &Tensor,
    name:   *const u8,

    /**
      | 1-indexed
      |
      */
    pos:    i32,
}

impl TensorArg {
    
    pub fn new(
        tensor: &Tensor,
        name:   *const u8,
        pos:    i32) -> Self {
    
        todo!();
        /*
        : tensor(tensor),
        : name(name),
        : pos(pos),

        
        */
    }
    
    pub fn operator_star(&self) -> &Tensor {
        
        todo!();
        /*
            return tensor;
        */
    }
}

impl Deref for TensorArg {

    type Target = Tensor;
    
    #[inline] fn deref(self) -> &Self::Target {
        todo!();
        /*
            return &tensor;
        */
    }
}


pub struct TensorGeometryArg {
    tensor: TensorGeometry,
    name:   *const u8,

    /**
      | 1-indexed
      |
      */
    pos:    i32,
}

impl TensorGeometryArg {
    
    pub fn new(arg: TensorArg) -> Self {
    
        todo!();
        /*


            : tensor(TensorGeometry{arg.tensor}), name(arg.name), pos(arg.pos)
        */
    }
    
    pub fn new(
        tensor: TensorGeometry,
        name:   *const u8,
        pos:    i32) -> Self {
    
        todo!();
        /*


            : tensor(tensor), name(name), pos(pos)
        */
    }
}

impl Deref for TensorGeometryArg {
    type Target = TensorGeometry;

    
    #[inline] fn deref(self) -> &Self::Target {
        todo!();
        /*
            return &tensor;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorUtils.cpp]

/**
  | The undefined convention: singular operators
  | assume their arguments are defined, but
  | functions which take multiple tensors will
  | implicitly filter out undefined tensors (to
  | make it easier to perform tests which should
  | apply if the tensor is defined, and should not
  | otherwise.)
  |
  | NB: This means that the n-ary operators take
  | lists of TensorArg, not TensorGeometryArg,
  | because the Tensor to TensorGeometry conversion
  | will blow up if you have undefined tensors.
  */
impl fmt::Display for TensorGeometryArg {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            if (t.pos == 0) {
        // 0 is distinguished; it usually indicates 'self' or the return
        // tensor
        out << "'" << t.name << "'";
      } else {
        out << "argument #" << t.pos << " '" << t.name << "'";
      }
      return out;
        */
    }
}

pub fn check_dim_a(
    c:      CheckedFrom,
    tensor: &Tensor,
    name:   *const u8,
    pos:    i32, // 1-indexed
    dim:    i64)  {

    todo!();
        /*
            TORCH_CHECK(
          tensor.dim() == dim,
          "Expected ",
          dim,
          "-dimensional tensor, but got ",
          tensor.dim(),
          "-dimensional tensor for ",
          TensorGeometryArg(TensorArg({tensor, name, pos})),
          " (while checking arguments for ",
          c,
          ")");
        */
}

pub fn check_dim_b(
    c:   CheckedFrom,
    t:   &TensorGeometryArg,
    dim: i64)  {

    todo!();
        /*
            TORCH_CHECK(t->dim() == dim,
        "Expected ", dim, "-dimensional tensor, but got ", t->dim(),
        "-dimensional tensor for ", t," (while checking arguments for ", c, ")");
        */
}

pub fn check_dim_range(
    c:         CheckedFrom,
    t:         &TensorGeometryArg,
    dim_start: i64,
    dim_end:   i64)  {

    todo!();
        /*
            TORCH_CHECK(
        t->dim() >= dim_start && t->dim() < dim_end,
        "Expected ", dim_start, " to ", (dim_end - 1), " dimensions, but got ",
        t->dim(), "-dimensional tensor for ", t, " (while checking arguments for ",
        c, ")");
        */
}


pub fn check_contiguous(
        c: CheckedFrom,
        t: &TensorGeometryArg)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t->is_contiguous(),
        "Expected contiguous tensor, but got non-contiguous tensor for ", t,
         " (while checking arguments for ", c, ")");
        */
}


pub fn check_all_contiguous(
        c:  CheckedFrom,
        ts: &[TensorArg])  {
    
    todo!();
        /*
            for (auto& t : ts) {
        if (!t->defined()) continue;
        checkContiguous(c, t);
      }
        */
}


pub fn check_size_a(
        c:     CheckedFrom,
        t:     &TensorGeometryArg,
        sizes: &[i32])  {
    
    todo!();
        /*
            checkDim(c, t, sizes.size());
      TORCH_CHECK(
        t->sizes().equals(sizes),
        "Expected tensor of size ", sizes, ", but got tensor of size ", t->sizes(),
        " for ", t, " (while checking arguments for ", c, ")");
        */
}

pub fn check_size_b(
        c:    CheckedFrom,
        t:    &TensorGeometryArg,
        dim:  i64,
        size: i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t->size(dim) == size,
        "Expected tensor to have size ", size, " at dimension ", dim,
        ", but got size ", t->size(dim), " for ", t,
        " (while checking arguments for ", c, ")");
        */
}


pub fn check_all_same(
        c:       CheckedFrom,
        tensors: &[TensorArg],
        fn_:     fn(
                _0: CheckedFrom,
                _1: &TensorArg,
                _2: &TensorArg
        ) -> c_void)  {
    
    todo!();
        /*
            const TensorArg* t0 = nullptr;
      for (auto& t : tensors) {
        if (!t->defined()) continue;
        if (t0 != nullptr) {
          fn(c, *t0, t);
        } else {
          t0 = &t;
        }
      }
        */
}

/**
  | capital is an amplifier of human labor
  |
  | -- robert breedlove
  |
  */
pub fn check_same_size(
        c:  CheckedFrom,
        t1: &TensorArg,
        t2: &TensorArg)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t1->sizes().equals(t2->sizes()),
        "Expected tensor for ", t1, " to have same size as tensor for ", t2,
        "; but ", t1->sizes(), " does not equal ", t2->sizes(),
        " (while checking arguments for ", c, ")");
        */
}


pub fn check_all_same_size(
        c:       CheckedFrom,
        tensors: &[TensorArg])  {
    
    todo!();
        /*
            checkAllSame(c, tensors, checkSameSize);
        */
}


pub fn check_numel(
        c:     CheckedFrom,
        t:     &TensorGeometryArg,
        numel: i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t->numel() == numel,
        "Expected tensor for ", t, " to have ", numel,
        " elements; but it actually has ", t->numel(), " elements",
        " (while checking arguments for ", c, ")");
        */
}

pub fn check_same_numel(
    c:  CheckedFrom,
    t1: &TensorArg,
    t2: &TensorArg)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t1->numel() == t2->numel(),
        "Expected tensor for ", t1,
        " to have same number of elements as tensor for ", t2, "; but ",
        t1->numel(), " does not equal ", t2->numel(),
        " (while checking arguments for ", c, ")");
        */
}

pub fn check_all_same_numel(
        c:       CheckedFrom,
        tensors: &[TensorArg])  {
    
    todo!();
        /*
            checkAllSame(c, tensors, checkSameNumel);
        */
}

pub fn check_same_gpu(
        c:  CheckedFrom,
        t1: &TensorArg,
        t2: &TensorArg)  {
    
    todo!();
        /*
            if (! (t1->is_cuda()) || ! (t2->is_cuda())) {
        ostringstream oss;
        if (! t1->is_cuda()) {
          oss << "Tensor for " << t1 << " is on CPU, ";
        }
        if (! t2->is_cuda()) {
          oss << "Tensor for " << t2 << " is on CPU, ";
        }
        oss << "but expected " << ((!(t1->is_cuda() || t2->is_cuda())) ? "them" : "it")
            << " to be on GPU (while checking arguments for " << c << ")";
        AT_ERROR(oss.str());
      }
      TORCH_CHECK(
        t1->get_device() == t2->get_device(),
        "Expected tensor for ", t1, " to have the same device as tensor for ", t2,
        "; but device ", t1->get_device(), " does not equal ", t2->get_device(),
        " (while checking arguments for ", c, ")");
        */
}


pub fn check_all_same_gpu(
        c:       CheckedFrom,
        tensors: &[TensorArg])  {
    
    todo!();
        /*
            checkAllSame(c, tensors, checkSameGPU);
        */
}


pub fn check_same_type(
        c:  CheckedFrom,
        t1: &TensorArg,
        t2: &TensorArg)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t1->options().type_equal(t2->options()),
        "Expected tensor for ", t1, " to have the same type as tensor for ", t2,
        "; but type ", t1->toString(), " does not equal ", t2->toString(),
        " (while checking arguments for ", c, ")");
        */
}


pub fn check_scalar_type(
        c:  CheckedFrom,
        t:  &TensorArg,
        ty: ScalarType)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t->scalar_type() == ty,
        "Expected tensor for ", t, " to have scalar type ", toString(ty),
        "; but got ", t->toString(), " instead (while checking arguments for ", c,
        ")");
        */
}


pub fn check_scalar_types(
        c: CheckedFrom,
        t: &TensorArg,
        l: &[ScalarType])  {
    
    todo!();
        /*
            if (find(l.begin(), l.end(), t->scalar_type()) == l.end()) {
          ostringstream oss;
          oss << "Expected tensor for " << t << " to have one of the following "
              << "scalar types: ";
          usize i = 0;
          for (auto ty : l) {
            if (i != 0) {
              oss << ", ";
            }
            oss << toString(ty);
            i++;
          }
          oss << "; but got " << t->toString()
              << " instead (while checking arguments for " << c << ")";
          AT_ERROR(oss.str());
        }
        */
}


pub fn check_all_same_type(
        c:       CheckedFrom,
        tensors: &[TensorArg])  {
    
    todo!();
        /*
            checkAllSame(c, tensors, checkSameType);
        */
}


pub fn check_same_dim(
        c:  CheckedFrom,
        t1: &TensorGeometryArg,
        t2: &TensorGeometryArg)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t1->dim() == t2->dim(),
        "Expected tensor for ", t1, " to have the same dimension as tensor for ",
        t2, "; but ", t1->dim(), " does not equal ", t2->dim(),
        " (while checking arguments for ", c, ")");
        */
}


pub fn check_defined(
        c: CheckedFrom,
        t: &TensorArg)  {
    
    todo!();
        /*
            TORCH_CHECK(
        t->defined(),
        "Expected tensor for ", t, " to be non-null, but it was undefined ",
        " (while checking arguments for ", c, ")");
        */
}


pub fn check_all_defined(
        c:  CheckedFrom,
        ts: &[TensorArg])  {
    
    todo!();
        /*
            // NB: don't filter defined here
      for (auto t : ts) {
        checkDefined(c, t);
      }
        */
}

pub fn check_backend(
    c:       CheckedFrom,
    t:       &Tensor,
    backend: Backend)  {

    todo!();
        /*
            TORCH_CHECK(
        !t.defined() || t.options().backend() == backend,
        "Expected tensor to have ", toString(backend),
        " Backend, but got tensor with ", toString(t.options().backend()), " Backend ",
        "(while checking arguments for ", c, ")");
        */
}

pub fn check_backend_plural(
    c:       CheckedFrom,
    tensors: &[Tensor],
    backend: Backend)  {
    
    todo!();
        /*
            for (auto &t : tensors) {
        checkBackend(c, t, backend);
      }
        */
}

pub fn check_device_type_a(
    c:           CheckedFrom,
    t:           &Tensor,
    device_type: DeviceType)  {
    
    todo!();
        /*
            TORCH_CHECK(
          !t.defined() || t.device().type() == device_type,
          "Expected tensor to have ", device_type,
          " DeviceType, but got tensor with ", t.device().type(), " DeviceType ",
          "(while checking arguments for ", c, ")");
        */
}

/**
  | a necessary component of the fiat currency
  | complex is financial illiteracy
  |
  | - quoted by robert breedlove 2/17/2023
  |
  */
pub fn check_device_type_b(
    c:           CheckedFrom,
    tensors:     &[Tensor],
    device_type: DeviceType)  {
    
    todo!();
        /*
            for (auto &t : tensors) {
        checkDeviceType(c, t, device_type);
      }
        */
}

pub fn check_layout_a(
    c:      CheckedFrom,
    t:      &Tensor,
    layout: Layout)  {
    
    todo!();
        /*
            TORCH_CHECK(
        !t.defined() || t.layout() == layout,
        "Expected tensor to have ", layout,
        " Layout, but got tensor with ", t.layout(), " Layout ",
        "(while checking arguments for ", c, ")");
        */
}

pub fn check_layout_b(
    c:       CheckedFrom,
    tensors: &[Tensor],
    layout:  Layout)  {

    todo!();
        /*
            for (auto &t : tensors) {
        checkLayout(c, t, layout);
      }
        */
}

pub fn maybe_data_ptr_a(tensor: &Tensor)  {
    
    todo!();
        /*
            return tensor.defined() ? (void *)tensor.data_ptr() : nullptr;
        */
}

pub fn maybe_data_ptr_b(tensor: &TensorArg)  {
    
    todo!();
        /*
            return tensor->defined() ? (void *)tensor->data_ptr() : nullptr;
        */
}

/**
  | See TensorUtils.h on why this is useful
  | now that we cache is_contiguous.
  |
  | Return if the tensor geometry represented by
  | `sizes` and `strides` is contiguous
  |
  | Although we cache is_contiguous in tensor now,
  | this is till useful because it allows checking
  | if a particular geometry is contiguous without
  | explicitly constructing a tensor, e.g., when
  | you want to choose a kernel strategy based on
  | whether a subgeometry is contiguous.
  */
pub fn geometry_is_contiguous(
        sizes:   &[i32],
        strides: &[i32]) -> bool {
    
    todo!();
        /*
            i64 dim = sizes.size();
      i64 expected_stride = 1;
      bool contig_if_nonempty = true;
      for (i64 i = dim - 1; i >= 0; i--) {
        if (sizes[i] == 0) {
          return true;
        }
        if (contig_if_nonempty) {
          if (sizes[i] != 1 && strides[i] != expected_stride) {
            contig_if_nonempty = false;
          }
          expected_stride *= sizes[i];
        }
      }
      return contig_if_nonempty;
        */
}

/**
  | Correspond to
  | THCUNN_check_dim_size/THNN_check_dim_size
  |
  */
pub fn check_dim_size(
        tensor:   &Tensor,
        dim:      i64,
        dim_size: i64,
        size:     i64)  {
    
    todo!();
        /*
            /* Check dimension size of a tensor */
      TORCH_CHECK(
          tensor.dim() == dim && tensor.size(dim_size) == size,
          "Expected a tensor of dimension ",
          dim,
          " and tensor.size[",
          dim_size,
          "] == ",
          size,
          " but got: dimension ",
          tensor.dim(),
          " and tensor.size[",
          dim_size,
          "] = ",
          tensor.size(dim_size));
        */
}

pub fn default_strides(sizes: &[i32]) -> Vec<i64> {
    
    todo!();
        /*
            vector<i64> strides(sizes.size());
      i64 stride = 1;
      for(usize i = sizes.size(); i > 0; --i) {
        strides[i-1] = stride;
        stride *= sizes[i-1];
      }
      return strides;
        */
}

pub fn compute_storage_nbytes(
        sizes:          &[i32],
        strides:        &[i32],
        itemsize_bytes: usize) -> usize {
    
    todo!();
        /*
            // size of the underlying storage is 1 bigger than the offset
      // of the last element according to stride
      usize size = 1;
      for(usize i = 0; i < sizes.size(); i++) {
        if(sizes[i] == 0) {
          return 0;
        }
        size += strides[i]*(sizes[i]-1);
      }
      return size * itemsize_bytes;
        */
}

/**
  | On a high level,
  |
  | 1. separate `oldshape` into chunks of
  |    dimensions, where the dimensions are
  |    ``contiguous'' in each chunk, i.e.,
  |    oldstride[i] = oldshape[i+1]
  |    * oldstride[i+1]
  |
  | 2. `newshape` must be able to be separated into
  |    same number of chunks as `oldshape` was
  |    separated into, where each chunk of newshape
  |    has matching ``numel'', i.e., number of
  |    subspaces, as the corresponding chunk of
  |    `oldshape`.
  |
  | templatized for DimVector and IntArrayRef use
  | cases, see overloads of computeStride() below.
  |
  */
#[inline] pub fn compute_stride_impl<ResultVec, NewShapeVec>(
        oldshape:  &[i32],
        oldstride: &[i32],
        newshape:  &NewShapeVec,
        to_result: fn(_0: &&[i32]) -> ResultVec) -> Option<ResultVec> {

    todo!();
        /*
            if (oldshape.empty()) {
        return ResultVec(newshape.size(), 1);
      }

      // NOTE: stride is arbitrary in the numel() == 0 case;
      // to match NumPy behavior we copy the strides if the size matches, otherwise
      // we use the stride as if it were computed via resize.
      // This could perhaps be combined with the below code, but the complexity
      // didn't seem worth it.
      const i64 numel = multiply_integers(oldshape);
      if (numel == 0 && oldshape.equals(newshape)) {
        return toResult(oldstride);
      }

      ResultVec newstride(newshape.size());
      if (numel == 0) {
        for (i64 view_d = newshape.size() - 1; view_d >= 0; view_d--) {
          if (view_d == (i64)(newshape.size() - 1)) {
            newstride[view_d] = 1;
          } else {
            newstride[view_d] =
              max<i64>(newshape[view_d+1], 1) * newstride[view_d+1];
          }
        }
        return newstride;
      }

      i64 view_d = (i64)newshape.size() - 1;
      // stride for each subspace in the chunk
      i64 chunk_base_stride = oldstride.back();
      // numel in current chunk
      i64 tensor_numel = 1;
      i64 view_numel = 1;
      for (i64 tensor_d = oldshape.size() - 1; tensor_d >= 0; tensor_d--) {
        tensor_numel *= oldshape[tensor_d];
        // if end of tensor size chunk, check view
        if ((tensor_d == 0) ||
            (oldshape[tensor_d - 1] != 1 &&
             oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
          while (view_d >= 0 &&
                (view_numel < tensor_numel || newshape[view_d] == 1)) {
            newstride[view_d] = view_numel * chunk_base_stride;
            view_numel *= newshape[view_d];
            view_d--;
          }
          if (view_numel != tensor_numel) {
            return nullopt;
          }
          if (tensor_d > 0) {
            chunk_base_stride = oldstride[tensor_d - 1];
            tensor_numel = 1;
            view_numel = 1;
          }
        }
      }
      if (view_d != -1) {
        return nullopt;
      }
      return newstride;
        */
}

pub fn compute_stride_a(
        oldshape:  &[i32],
        oldstride: &[i32],
        newshape:  &[i32]) -> Option<Vec<i64>> {
    
    todo!();
        /*
            auto toResult = [](const IntArrayRef& a) { return a.vec(); };
      return computeStride_impl<vector<i64>, IntArrayRef>(oldshape, oldstride, newshape, toResult);
        */
}

pub fn compute_stride_b(
        oldshape:  &[i32],
        oldstride: &[i32],
        newshape:  &DimVector) -> Option<DimVector> {
    
    todo!();
        /*
            auto toResult = [](const IntArrayRef& a) { return DimVector(a); };
      return computeStride_impl<DimVector, DimVector>(oldshape, oldstride, newshape, toResult);
        */
}
