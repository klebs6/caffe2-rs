crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/IndexingUtils.h]

pub fn invalid_mask(
        self_:    &Tensor,
        idx:      i64,
        mask:     &Tensor,
        mask_idx: i64)  {
    
    todo!();
        /*
            TORCH_CHECK_INDEX(false, "The shape of the mask ", mask.sizes(), " at index ", maskIdx,
      " does not match the shape of the indexed tensor ", self.sizes(), " at index ", idx);
        */
}

pub fn expand_tensors(
        self_:   &Tensor,
        indices: &TorchList<Option<Tensor>>) -> Vec<Tensor> {
    
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

pub fn check_index_tensor_types(indices: &TorchList<Option<Tensor>>)  {
    
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

#[inline] pub fn to_list_of_optional_tensors_a(list: &[Tensor]) -> TorchList<Option<Tensor>> {
    
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

#[inline] pub fn to_list_of_optional_tensors_b(list: &[IValue]) -> TorchList<Option<Tensor>> {
    
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

pub fn has_contiguous_subspace(tl: TensorList) -> bool {
    
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
        indices: TensorList) -> (Tensor,Vec<Tensor>) {
    
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
        indices: TensorList) -> (Tensor,Vec<Tensor>,Vec<i64>) {
    
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
        src:     &Tensor,
        indices: TensorList) -> Self {
    
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/IndexingUtils.cpp]

pub fn can_use_32bit_index_math(
        t:        &Tensor,
        max_elem: Option<i64>) -> bool {

    let max_elem: i64 = max_elem.unwrap_or(i32_max);
    
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
