crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/NamedTensorUtils.h]

pub type NameVector = SmallVector<Dimname,kDimVectorStaticSize>;

#[inline] pub fn has_names(tensors: TensorList) -> bool {
    
    todo!();
        /*
            return any_of(
          tensors.begin(), tensors.end(), [](const Tensor& t) { return t.has_names(); });
        */
}

#[inline] pub fn report_nyi_dimname_overload(op_name: *const u8)  {
    
    todo!();
        /*
            TORCH_CHECK(
          false,
          op_name, ": You passed a dimname (string) to this op in place of a dimension "
          "index but it does not yet support this behavior. Please pass a dimension "
          "index to work around this.");
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/NamedTensorUtils.cpp]

/// Returns "Tensor['N', 'C', 'H', 'W']" for
/// a tensor with names ('N', 'C', 'H', 'W').
///
pub fn to_dimname_repr(tensor: &Tensor) -> String {
    
    todo!();
        /*
            ostringstream os;
      os << "Tensor" << tensor.names();
      return os.str();
        */
}

/**
  | Converts dim to an positional index.
  | Errors if `dim` cannot be used to refer
  | to any dimension of tensor.
  |
  */
pub fn dimname_to_position(
        tensor: &Tensor,
        dim:    Dimname) -> i64 {
    
    todo!();
        /*
            TORCH_CHECK(dim.type() != NameType::WILDCARD,
          "Please look up dimensions by name, got: name = None.");
      TORCH_CHECK(tensor.has_names(),
          "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");
      const auto names = tensor.names();

      const auto it = find(names.begin(), names.end(), dim);
      TORCH_CHECK(it != names.end(),
          "Name ", dim, " not found in ", toDimnameRepr(tensor), ".");

      return distance(names.begin(), it);
        */
}

pub fn dimnames_to_positions(
        tensor: &Tensor,
        dims:   DimnameList) -> Vec<i64> {
    
    todo!();
        /*
            vector<i64> result;
      result.reserve(dims.size());
      for (const auto& name : dims) {
        result.push_back(dimname_to_position(tensor, name));
      }
      return result;
        */
}

pub fn report_positional_error(
        name:        &Dimname,
        other_name:  &Dimname,
        names:       DimnameList,
        other_names: DimnameList,
        action:      *const u8)  {
    
    todo!();
        /*
            // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
      TORCH_CHECK(false,
          "Error when attempting to ", action, " dims ", names, " and dims ",
          other_names, ": dim ", name, " and dim ", other_name, " are at the same position "
          "from the right but do not match.")
        */
}

pub fn check_for_misalignment(
        name:        &Dimname,
        names:       DimnameList,
        other_names: DimnameList,
        action:      *const u8)  {
    
    todo!();
        /*
            if (name.isWildcard()) {
        return;
      }
      auto it = find(other_names.begin(), other_names.end(), name);
      // TODO(zou3519): Can improve message by checking if names are alignable and suggesting workarounds
      TORCH_CHECK(it == other_names.end(),
          "Misaligned dims when attempting to ", action, " dims ", names, " and dims ",
          other_names, ": dim ", name, " appears in a different position from the right "
          "across both lists.");
        */
}

/**
  | Assumption: A DimnameList can have
  | no duplicate full names with the exception
  | of wildcards
  |
  | Unifies two DimnameList to produce
  | a third. This is useful for implementing the
  | named inference rule for binary broadcasting
  | operations like add.
  |
  | There are three main constraints:
  |
  | 1) Check matching: Names must match
  | positionally from the right.
  |
  | 2) Check misaligned: If a name `n` is in
  |    `names`, then it must appear at the same
  |    index from the right in other.
  |
  | 3) The output names are obtained by unifying
  | the names individually from the right.
  |
  */
pub fn unify_from_right(
    names:       DimnameList,
    other_names: DimnameList,
    action:      Option<&'static str>) -> Vec<Dimname> {

    let action = action.unwrap_or("broadcast");
    
    todo!();
        /*
            const auto wildcard = Dimname::wildcard();
      const auto size = max(names.size(), other_names.size());
      auto result = vector<Dimname>(size, wildcard);

      auto names_it = names.rbegin();
      auto other_it = other_names.rbegin();
      auto result_it = result.rbegin();
      while (names_it != names.rend() || other_it != other_names.rend()) {
        const auto& name = names_it == names.rend() ? wildcard : *names_it;
        const auto& other_name = other_it == other_names.rend() ? wildcard : *other_it;

        // Step 1: Check that the names match
        const auto maybeName = name.unify(other_name);
        if (!maybeName) {
          report_positional_error(name, other_name, names, other_names, action);
        }
        *result_it = *maybeName;

        // Step 2: Check that the names are not misaligned
        if (!name.isBasic() || !other_name.isBasic()) {
          // Let: N = max(len(names), len(other_names))
          //      K = # of special names among names and other_names.
          // This search (including the outer loop) is O(N*K) but typically # of dims is small.
          check_for_misalignment(name, names, other_names, action);
          check_for_misalignment(other_name, other_names, names, action);
        }

        if (names_it != names.rend()) {
          ++names_it;
        }
        if (other_it != other_names.rend()) {
          ++other_it;
        }
        ++result_it;
      }
      return result;
        */
}

/**
  | [NOTE] Writing name inference rules
  |
  | Operators that support named tensors are either
  | composed of operations that support named
  | tensors or implement some name inference
  | rule. An op that implements its own name
  | inference rule generally looks like the
  | following:
  |
  | Tensor op(...) {
  |   perform_shape_checks(...);
  |   # (1)
  |   auto maybe_outnames = compute_outnames(...);
  |   auto result = [&]() {
  |     NoNamesGuard guard;
  |     return op_impl(...);
  |   }();
  |   # (2)
  |   propagate_names_if_nonempty(result, maybe_outnames);
  |
  | Each op has (1) a compute outnames step and (2)
  | a propagate names step.
  |
  | compute_outnames is responsible for checking
  | that input names match and determining what the
  | output names should be.
  |
  | It returns either:
  |
  | - {} (if the inputs tensors are all unnamed)
  |
  | - non-empty outnames.
  |
  | propagate_names_if_nonempty propagates the
  | outnames if they exist to the result tensors.
  |
  | The {} case is an optimization; if the user
  | does not use named tensors they pay no perf
  | cost for it.
  */
pub mod namedinference {

    use super::*;

    pub fn compute_included_idxs(
            excluded_idxs: &[i32],
            ndims:         i64) -> BitSet<DimBitsetSize> {
        
        todo!();
            /*
                auto result = dim_list_to_bitset(excluded_idxs, ndims);
              result.flip();
              return result;
            */
    }

    pub fn assert_names_equal(
            a: DimnameList,
            b: DimnameList)  {
        
        todo!();
            /*
                TORCH_CHECK(a == b,
                  "Name mismatch: specified out tensor with names ", a,
                  " are not the same as the computed output names ", b,
                  ". Please rename the out tensor's dims with `Tensor.rename`.");
            */
    }

    /**
      | Propagates `names` to `result` if `names`
      | is not empty.
      |
      | `names` can be empty; see [NOTE] Writing
      | name inference rules
      |
      | If `names` is not empty, `names.size()`
      | should equal `result.dim()`.
      |
      | When in doubt, use this overload instead of
      | the others.
      |
      */
    pub fn propagate_names_if_nonempty(
            result:         &Tensor,
            maybe_names:    DimnameList,
            validate_names: bool) -> &Tensor {

        let validate_names: bool = validate_names.unwrap_or(false);

        /*
        /// TensorImpl* overloads for Legacy TH/THC
        /// code. Use these sparingly.
        ///
        pub fn propagate_names_if_nonempty(
            result:         *mut TensorImpl,
            maybe_names:    DimnameList,
            validate_names: bool) -> *mut TensorImpl {

            let validate_names: bool = validate_names.unwrap_or(false);
            
            todo!();
                /*
                    if (maybe_names.empty()) {
                    return result;
                  }
                  return propagate_names(result, maybe_names, validate_names);
                */
        }
        */

        todo!();
            /*
                propagate_names_if_nonempty(result.unsafeGetTensorImpl(), maybe_names, validate_names);
              return result;
            */
    }

    /**
      | Propagates `names` to `result`. Only use
      | this if we are certain that there are names
      | to propagate (that names is not empty).
      |
      */
    pub fn propagate_names(
            result:         &Tensor,
            names:          DimnameList,
            validate_names: bool) -> &Tensor {

        let validate_names: bool = validate_names.unwrap_or(false);

        /*
        pub fn propagate_names(
                result:         *mut TensorImpl,
                names:          DimnameList,
                validate_names: bool) -> *mut TensorImpl {
            
            todo!();
                /*
                    if (result->dim() > 0) {
                    TORCH_INTERNAL_ASSERT(
                        !names.empty(),
                        "propagate_names: passed in empty names to propagate to result with",
                        " shape ", result->sizes(), ". Empty names means that name inference did",
                        "not occur; use `propagate_names_if_nonempty` instead of `propagate_names`.");
                  }
                  if (!has_names(result)) {
                    internal_set_names_inplace(result, names, validate_names);
                  } else {
                    assert_names_equal(get_names(result), names);
                  }
                  return result;
                */
        }
        */
            
        todo!();
            /*
                propagate_names(result.unsafeGetTensorImpl(), names, validate_names);
              return result;
            */
    }

    /// Propagates all names except for those at
    /// the excluded_idxs.
    ///
    pub fn propagate_names_except(
        result:        &Tensor,
        src:           &Tensor,
        excluded_idxs: &[i32])  {
        
        todo!();
            /*
                if (!result.has_names() && !src.has_names()) {
                return;
              }
              auto src_names = src.names();
              auto result_dim = result.dim();
              auto src_dim = src_names.size();
              TORCH_INTERNAL_ASSERT(src_dim - excluded_idxs.size() == result_dim);

              // fast path
              if (excluded_idxs.size() == 1) {
                vector<Dimname> outnames = src_names.vec();
                outnames.erase(outnames.begin() + maybe_wrap_dim(excluded_idxs[0], src_dim));
                propagate_names(result, outnames);
                return;
              }

              vector<Dimname> outnames;
              outnames.reserve(result_dim);
              auto included_idxs = compute_included_idxs(excluded_idxs, src_dim);
              for (usize dim = 0; dim < src_dim; ++dim) {
                if (included_idxs[dim]) {
                  outnames.push_back(src_names[dim]);
                }
              }
              propagate_names(result, outnames);
            */
    }

    /// Used for reduction ops that have
    /// a `keepdim` arg.
    ///
    pub fn propagate_names_for_reduction(
        result:       &Tensor,
        src:          &Tensor,
        reduced_dims: &[i32],
        keepdim:      bool)  {

        todo!();
            /*
                if (keepdim) {
                propagate_names(result, src);
                return;
              }
              // This actually means "full reduction"
              if (reduced_dims.size() == 0) {
                return;
              }
              propagate_names_except(result, src, reduced_dims);
            */
    }

    /// Propagates all names from src to result.
    ///
    pub fn propagate_names_from_src_to_result(
            result: &Tensor,
            src:    &Tensor)  {
        
        /*
        pub fn propagate_names(
                result: *mut TensorImpl,
                src:    *mut TensorImpl)  {
            
            todo!();
                /*
                    if (result == src) {
                    return;
                  }
                  if (!has_names(result) && !has_names(src)) {
                    return;
                  }
                  propagate_names(result, get_names(src));
                */
        }
        */

        todo!();
            /*
                propagate_names(result.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            */
    }

    pub fn compute_squeeze_outnames(tensor: &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!tensor.has_names()) {
                return {};
              }
              vector<Dimname> outnames;
              auto tensor_names = tensor.names();
              for (i64 d = 0; d < tensor.dim(); d++) {
                if (tensor.sizes()[d] != 1) {
                  outnames.push_back(tensor_names[d]);
                }
              }
              return outnames;
            */
    }

    pub fn compute_diagonal_outnames(
        tensor: &Tensor,
        dim1:   i64,
        dim2:   i64) -> Vec<Dimname> {

        todo!();
            /*
                if (!tensor.has_names()) {
                return {};
              }
              vector<Dimname> outnames;
              auto tensor_names = tensor.names();
              for (i64 d = 0; d < tensor.dim(); d++) {
                if (d == dim1 || d == dim2) {
                  continue;
                }
                outnames.push_back(tensor_names[d]);
              }
              outnames.push_back(Dimname::wildcard());
              return outnames;
            */
    }

    /**
      | tensor_dotted_dim and other_dotted_dim are
      | the dimensions of the two tensors that we
      | contract together. Usually other_dotted_dim
      | is 0 and tensor_dotted_dim is the last dim
      | of tensor, but there are some special cases
      | like einsum and tensordot where one can
      | contract arbitrary dims.
      |
      */
    pub fn compute_dot_product_outnames(
        tensor_names:      DimnameList,
        tensor_dotted_dim: i64,
        other_names:       DimnameList,
        other_dotted_dim:  i64) -> Vec<Dimname> {
        
        todo!();
            /*
                i64 num_outnames = tensor_names.size() + other_names.size() - 2;
              if (num_outnames == 0) {
                return {};
              }
              vector<Dimname> outnames(num_outnames, Dimname::wildcard());
              i64 index = 0;
              for (usize j = 0; j < tensor_names.size(); ++j) {
                if (j == tensor_dotted_dim) continue;
                outnames[index++] = tensor_names[j];
              }
              for (usize j = 0; j < other_names.size(); ++j) {
                if (j == other_dotted_dim) continue;
                outnames[index++] = other_names[j];
              }
              return outnames;
            */
    }

    pub fn check_feature_names_are_distinct(
        self_names:  DimnameList,
        other_names: DimnameList,
        outnames:    DimnameList)  {
        
        todo!();
            /*
                if (self_names.size() < 2 || other_names.size() < 2) {
                // There are less than 2 feature dims in outnames so there is nothing to check
                return;
              }
              auto feature0 = outnames[outnames.size() - 2];
              auto feature1 = outnames[outnames.size() - 1];
              TORCH_CHECK(
                feature0 == Dimname::wildcard() || feature0 != feature1,
                "Matrix multiplying Tensor", self_names,
                " with Tensor", other_names,
                " would produce output tensor with duplicate names ",
                outnames,
                ". Please rename the input tensors with `Tensor.rename` to prevent this.");
            */
    }

    pub fn batch_dims(names: DimnameList) -> DimnameList {
        
        todo!();
            /*
                if (names.size() <= 2) {
                return {};
              }
              return DimnameList(names.begin(), names.end() - 2);
            */
    }

    pub fn feature_dims(names: DimnameList) -> DimnameList {
        
        todo!();
            /*
                if (names.size() <= 2) {
                return names;
              }
              return DimnameList(names.end() - 2, 2);
            */
    }

    pub fn are_distinct(
        batch_dims:   DimnameList,
        feature_dims: DimnameList) -> bool {
        
        todo!();
            /*
                for (const auto& target : feature_dims) {
                if (target.isWildcard()) {
                  continue;
                }
                if (any_of(batch_dims.begin(), batch_dims.end(),
                      [&](const Dimname& dim) { return target == dim; })) {
                  return false;
                }
              }
              return true;
            */
    }

    pub fn num_batch_dims(names: DimnameList) -> i64 {
        
        todo!();
            /*
                if (names.size() <= 2) {
                return 0;
              }
              return names.size() - 2;
            */
    }

    pub fn compute_matmul_outnames_with_dimname_list(
        self_names:  DimnameList,
        other_names: DimnameList) -> Vec<Dimname> {

        todo!();
            /*
                TORCH_CHECK(self_names.size() >= 1 && other_names.size() >= 1,
                  "both arguments to matmul need to be at least 1D, but they are ",
                  self_names.size(), "D and ", other_names.size(), "D");

              // matmul performs a batch matrix multiply between self and other, each of which
              // can either be:
              // - a batches of matrices (if dim > 2)
              // - a matrix (if dim == 2)
              // - a vector (if dim == 1)
              //
              // To compute output names, we unify the batch dimensions because those are
              // broadcastable to get the output batch dimensions.
              //
              // After that, we append some names that are equal to the result of the matmul
              // without batch dimensions. Those names are computed by removing the names
              // of the dimensions that were contracted away. We always contract the
              // last dim of the first tensor with the first feature dimension of the second.

              // Get the output's batch dimension names
              auto wrapped_self_names = TensorNames(self_names, 0, num_batch_dims(self_names));
              const auto wrapped_other_names = TensorNames(other_names, 0, num_batch_dims(other_names));
              auto& working_names = wrapped_self_names.unifyFromRightInplace(wrapped_other_names, "matmul");

              // Append the result of each individual (non-batched) matmul.
              // If either of self or other have dim 1, that means they are a vector. Vectors get
              // completely contracted away during matmul so we don't take any names from them.
              if (self_names.size() >= 2) {
                working_names.append(TensorName(self_names, -2));
              }
              if (other_names.size() >= 2) {
                working_names.append(TensorName(other_names, -1));
              }
              const auto result = working_names.toDimnameVec();

              check_feature_names_are_distinct(self_names, other_names, result);
              return result;
            */
    }

    pub fn propagate_names_for_addmv(
        mat:  &Tensor,
        vec:  &Tensor,
        bias: &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!mat.has_names() &&
                  !vec.has_names() && !bias.has_names()) {
                return vector<Dimname>{};
              }
              auto mv_outnames = compute_matmul_outnames(mat.names(), vec.names());
              return unify_from_right(mv_outnames, bias.names());
            */
    }

    /// result = m1 @ m2 + bias
    ///
    pub fn propagate_names_for_addmm(
        m1:   &Tensor,
        m2:   &Tensor,
        bias: &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!m1.has_names() && !m2.has_names() &&
                  !bias.has_names()) {
                return vector<Dimname>{};
              }

              auto mm_outnames = compute_matmul_outnames(m1.names(), m2.names());
              return unify_from_right(mm_outnames, bias.names());
            */
    }

    pub fn check_names_for_dot(
        vec1: *mut TensorImpl,
        vec2: *mut TensorImpl)  {
        
        todo!();
            /*
                if (!has_names(vec1) && !has_names(vec2)) {
                return;
              }
              compute_matmul_outnames(get_names(vec1), get_names(vec2));
            */
    }

    /**
      | expand adds new None dimensions.
      |
      | This is consistent with name inference
      | rules for binary ops that expect the named
      | dims to line up positionally from the
      | right. i.e.,
      |
      | Tensor[H, W].expand(3, 3, 3, 3) ->
      | Tensor[None, None, H, W]
      */
    pub fn propagate_names_for_expand(
        result: &Tensor,
        self_:  &Tensor)  {
        
        todo!();
            /*
                if (!self.has_names()) {
                return;
              }
              auto result_dim = result.dim();
              if (self.dim() == result_dim) {
                propagate_names(result, self);
                return;
              }
              vector<Dimname> outnames(result_dim, Dimname::wildcard());
              copy(
                  self.opt_names()->begin(),
                  self.opt_names()->end(),
                  outnames.begin() + result_dim - self.dim());
              propagate_names(result, outnames);
            */
    }

    pub fn compute_broadcast_outnames(
        self_: &Tensor,
        other: &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!self.has_names() && !other.has_names()) {
                return {};
              }
              return unify_from_right(self.names(), other.names());
            */
    }

    pub fn broadcast_to_outnames(
        tensor:           &Tensor,
        reference_tensor: &Tensor,
        op_name:          *const u8) -> Vec<Dimname> {

        todo!();
            /*
                if (!tensor.has_names() && !reference_tensor.has_names()) {
                return {};
              }
              auto reference_names = reference_tensor.names();
              auto tensor_names = tensor.names();
              TORCH_CHECK(
                  reference_names.size() >= tensor_names.size(),
                  op_name, ": attempted to broadcast Tensor", tensor_names, " to Tensor",
                  reference_names, " but the number of dims (", tensor_names.size(),
                  ") must be less than or equal to the number of dims in the tensor (",
                  reference_names.size(), ")");
              return unify_from_right(reference_names, tensor_names);
            */
    }

    /// he is dead - even in this world - he who
    /// has no belief in another.
    ///
    /// -Goethe
    ///
    pub fn compute_cat_outnames(tensors: TensorList) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!has_names(tensors)) {
                return {};
              }
              vector<Dimname> result;
              for (const auto& tensor : tensors) {
                const auto tensor_names = tensor.names();
                TORCH_CHECK(tensor_names.size() > 0, "zero-dimensional tensor cannot be concatenated");
                TORCH_CHECK(result.empty() || tensor_names.size() == result.size(),
                    "Tensors must have same number of dimensions: got ", result.size(),
                    " and ", tensor_names.size());
                result = unify_from_right(result, tensor_names, "cat");
              }
              return result;
            */
    }

    pub fn compute_matmul_outnames(
            self_: &Tensor,
            other: &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!self.has_names() && !other.has_names()) {
                return {};
              }
              return compute_matmul_outnames(self.names(), other.names());
            */
    }

    pub fn compute_cdist_outnames(
            self_: &Tensor,
            other: &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!self.has_names() && !other.has_names()) {
                return {};
              }
              const auto self_names = self.names();
              const auto other_names = other.names();

              auto self_batch = TensorNames(self_names, 0, num_batch_dims(self_names));
              const auto other_batch = TensorNames(other_names, 0, num_batch_dims(other_names));

              auto& result = self_batch.unifyFromRightInplace(other_batch, "cdist");

              // cdist treats self and other like batches of M x D and N X D tensors, respectively.
              // It computes the pairwise distance between each of the M vectors (of size D)
              // in `self` and each of the N vectors in `other`, returning a batch of M x N
              // distance values. We propagate the names of the dimension of size M (in self)
              // and the dimension of size N (in other), both of which are second-from-last.
              result.append(TensorName(self_names, -2));
              result.append(TensorName(other_names, -2));
              result.checkUnique("cdist");

              return result.toDimnameVec();
            */
    }

    pub fn compute_bmm_outnames(
            result: &mut Tensor,
            self_:  &Tensor,
            other:  &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!result.has_names() && !self.has_names() && !other.has_names()) {
                return {};
              }
              return compute_matmul_outnames(self.names(), other.names());
            */
    }

    pub fn compute_baddbmm_outnames(
            result: &mut Tensor,
            self_:  &Tensor,
            other:  &Tensor,
            bias:   &Tensor) -> Vec<Dimname> {
        
        todo!();
            /*
                if (!result.has_names() && !self.has_names()
                && !other.has_names() && !bias.has_names()) {
                return {};
              }
              auto bmm_names = compute_matmul_outnames(self.names(), other.names());
              auto baddbmm_names = unify_from_right(bias.names(), bmm_names);
              return baddbmm_names;
            */
    }

    pub fn are_names_equal(
            self_: *mut TensorImpl,
            other: *mut TensorImpl) -> bool {
        
        todo!();
            /*
                if (!has_names(self) && !has_names(other)) {
                return true;
              }
              return get_names(self) == get_names(other);
            */
    }
}
