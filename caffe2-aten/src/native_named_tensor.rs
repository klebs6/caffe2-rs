crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/NamedTensor.cpp]

pub fn rename_mut(
    self_: &mut Tensor,
    names: Option<DimnameList>) -> &mut Tensor {
    
    todo!();
        /*
            internal_set_names_inplace(self, names);
      return self;
        */
}

pub fn rename(
    self_: &Tensor,
    names: Option<DimnameList>) -> Tensor {

    todo!();
        /*
            auto result = self.alias();
      internal_set_names_inplace(result, names);
      return result;
        */
}

pub fn report_moving_unnamed_dim_error(
    names:                   DimnameList,
    other:                   DimnameList,
    is_aligning_two_tensors: bool)  {

    todo!();
        /*
            if (is_aligning_two_tensors) {
        TORCH_CHECK(false,
            "Aligning Tensor", names, " and Tensor", other,
            " would change the absolute position from the right of an unnamed dimension. ",
            "Please name unnamed dimensions to avoid ambiguity.");
      } else {
        TORCH_CHECK(false,
            "Aligning Tensor", names, " to `names` ", other,
            " would change the absolute position from the right of an unnamed dimension. ",
            "Please name unnamed dimensions to avoid ambiguity.");
      }
        */
}

pub fn report_not_a_subsequence_error(
    names:                   DimnameList,
    other:                   DimnameList,
    is_aligning_two_tensors: bool)  {
    
    todo!();
        /*
            if (is_aligning_two_tensors) {
        auto shorter = names.size() > other.size() ? other : names;
        auto longer = names.size() > other.size() ? names : other;
        TORCH_CHECK(false,
            "Could not align Tensor", shorter, " and Tensor", longer,
            " because ", shorter, " is not a subsequence of ", longer, ". ");
      } else {
        TORCH_CHECK(false,
            "Could not align Tensor", names, " to `names` ", other,
            " because ", names, " is not a subsequence of `names`.");
      }
        */
}

/**
  | Let tensor `t` have size `tensor_sizes` and
  | `tensor_names`.
  |
  | This helper function computes the resulting
  | size of `t` after aligning it to
  | `aligned_names`. Enforces the alignment rules
  | in Note [Alignment rules].
  |
  */
pub fn aligned_size(
    tensor_sizes:            &[i32],
    tensor_names:            DimnameList,
    aligned_names:           DimnameList,
    is_aligning_two_tensors: bool) -> Vec<i64> {
    
    todo!();
        /*
            vector<i64> expanded_sizes(aligned_names.size(), 1);
      ptrdiff_t dim = (ptrdiff_t)tensor_sizes.size() - 1;
      ptrdiff_t idx = (ptrdiff_t)aligned_names.size() - 1;
      for (; idx >= 0 && dim >= 0; --idx) {
        if (tensor_names[dim] != aligned_names[idx]) {
          continue;
        }
        // We've found a None name in `shorter` and `longer`. If their absolute positions
        // from the right are not equal, then aligning the two names would require
        // changing the absolute position from right of one of the None names,
        // violating condition 2 of our [Alignment rules].
        //
        // For example:
        // *, c, a, b
        //       *, a
        // [*, a] is a subsequence of [*, c, a, b], but in order to align them,
        // we'd have to move the * to create [*, c: 1, a, b: 1]
        if (tensor_names[dim].isWildcard() &&
            tensor_sizes.size() - dim != aligned_names.size() - idx) {
          report_moving_unnamed_dim_error(
              tensor_names, aligned_names, /*is_aligning_two_tensors=*/false);
        }
        expanded_sizes[idx] = tensor_sizes[dim];
        --dim;
      }
      if (dim != -1) {
        report_not_a_subsequence_error(
            tensor_names, aligned_names, /*is_aligning_two_tensors=*/false);
      }

      return expanded_sizes;
        */
}

pub fn refine_names(
    self_: &Tensor,
    names: DimnameList) -> Tensor {
    
    todo!();
        /*
            const auto self_names = self.names();
      TORCH_CHECK(self_names.size() == names.size(),
          "refine_names: cannot coerce Tensor", self_names, " to Tensor", names,
          " because they have a different number of dims (",
          self_names.size(), " and ", names.size(), " respectively).");
      check_names_valid_for(self, names);

      for (usize idx = 0; idx < self_names.size(); idx++) {
        const auto& self_name = self_names[idx];
        const auto& out_name = names[idx];
        if (self_name == out_name || self_name.isWildcard()) {
          continue;
        }
        if (out_name.isWildcard()) {
          TORCH_CHECK(false,
              "refine_names: cannot coerce Tensor", self_names, " to Tensor", names,
              " because ", self_name, " is more specific than ", out_name, " at index ",
              idx);
        }
        TORCH_CHECK(false,
            "refine_names: cannot coerce Tensor", self_names, " to Tensor", names,
            " because ", self_name, " is different from ", out_name, " at index ",
            idx);
        TORCH_INTERNAL_ASSERT(false); // done handling errors
      }

      auto result = self.alias();
      internal_set_names_inplace(result, names);
      return result;
        */
}

/**
  | [Alignment rules]
  | 
  | Aligns `tensor` to names with the following
  | rules:
  | 
  | - 1) Check that tensor.names is a subsequence
  | (not necessarily contiguous) of `names`.
  | 
  | - 2) Aligning tensor.names to names
  | must not change the absolute position
  | from the right of any unnamed dimension.
  | is_aligning_two_tensors tunes the
  | error message to better match the following
  | cases:
  | 
  | - 1) tensor.align_to(names) (is_aligning_two_tensors=false)
  | 
  | - 2) torch.align_tensors([tensor,
  | other]) (is_aligning_two_tensors=true)
  |
  */
pub fn align(
    tensor:                  &Tensor,
    names:                   DimnameList,
    is_aligning_two_tensors: bool) -> Tensor {
    
    todo!();
        /*
            vector<i64> expanded_sizes = aligned_size(
            tensor.sizes(),
            tensor.names(),
            names,
            is_aligning_two_tensors);
      auto result = tensor.rename(nullopt).view(expanded_sizes);
      internal_set_names_inplace(result, names);
      return result;
        */
}

pub fn count_unset(
    set:       BitSet<MaxNamedTensorDim>,
    up_to_idx: i64) -> i64 {

    todo!();
        /*
            i64 result = 0;
      for (const auto i : irange(up_to_idx)) {
        if (!set.test(i)) result++;
      }
      return result;
        */
}

/**
  | Handles `tensor.align_to(*order)` in the case
  | where there is an ellipsis.
  |
  | Let tensor: Tensor[N, C, H, W]. Consider
  | `tensor.align_to('W', ..., 'N')`
  |
  | We expand the `...` to "all unmentioned
  | dimensions, in the order which they appear in
  | the original tensor."
  |
  | `order` is passed in **without** the ellipsis
  | name. This is because ellipsis is not a valid
  | name in cpp right now. Future work should be
  | done on making ellipsis a valid name.
  |
  | `ellipsis_idx` is where the ellipsis occurs in
  | the Python call.
  |
  | In our example, `tensor.align_to('W', ...,
  | 'N')`, order = ['W', 'N'] and ellipsis_idx = 1.
  |
  */
pub fn align_to_a(
    tensor:       &Tensor,
    order:        DimnameList,
    ellipsis_idx: i64) -> Tensor {
    
    todo!();
        /*
            const auto tensor_names = tensor.names();
      const auto tensor_sizes = tensor.sizes();
      const auto tensor_strides = tensor.strides();
      const auto tensor_dim = tensor.sizes().size();
      constexpr i64 not_found = -1;

      // General strategy.
      //
      // Step 1: We compute the following 3 things:
      // 1. How many names the ellipsis should expand to
      // 2. Which names in `tensor.names` are not mentioned in `order`.
      // 3. Where names in `order` occur in tensor, if at all.
      //
      // Step 2: Compute the new sizes/strides/names.
      // First, determine the ndim of the output tensor (this is not obvious)
      // by counting the number of names in `tensor` that are not in `order`.
      // Next, fill in output sizes/strides/names by using `order` and knowledge
      // of which dimensions in `tensor` are unmentioned in `order`.

      bitset<kMaxNamedTensorDim> order_has_tensor_name;

      // tensor_idx_for[i] = j means that the ith name in `order`
      // appears in the jth element of tensor.
      vector<i64> tensor_idx_for(order.size(), not_found);

      for (const auto order_idx : irange(order.size())) {
        const auto name = order[order_idx];
        TORCH_CHECK(name.isBasic(),
            "align_to: the desired order of dimensions cannot contain a None name, got ",
            order);
        auto it = find(tensor_names.begin(), tensor_names.end(), name);
        if (it == tensor_names.end()) {
          continue;
        }
        auto idx_in_tensor = distance(tensor_names.begin(), it);
        tensor_idx_for[order_idx] = idx_in_tensor;
        order_has_tensor_name.set(idx_in_tensor);
      }

      const auto num_ellipsis_names = countUnset(order_has_tensor_name, tensor_dim);
      const auto out_dim = num_ellipsis_names + order.size();

      // Step 2: Now that we know the size of the output tensor, we can use the
      // metadata obtained from Step 1 to fill in the new sizes/strides/names
      vector<i64> new_sizes(out_dim, 1);
      vector<i64> new_strides(out_dim, 0);
      vector<Dimname> new_names(out_dim, Dimname::wildcard());

      auto setNewSizesStridesNamesFor = [&](i64 out_dim, i64 tensor_dim) {
        new_sizes[out_dim] = tensor_sizes[tensor_dim];
        new_strides[out_dim] = tensor_strides[tensor_dim];
        new_names[out_dim] = tensor_names[tensor_dim];
      };

      // Fill in the non-ellipsis dimensions
      for (auto order_idx = 0U; order_idx < order.size(); ++order_idx) {
        auto out_idx = order_idx;
        if (order_idx >= ellipsis_idx) {
          out_idx = order_idx + num_ellipsis_names;
        }
        const auto tensor_idx = tensor_idx_for[order_idx];
        if (tensor_idx == not_found) {
          // We are adding a new size-one dimension
          new_names[out_idx] = order[order_idx];
          continue;
        }
        setNewSizesStridesNamesFor(out_idx, tensor_idx);
      }

      // Fill in the ellipsis dimensions
      for (const auto tensor_idx : irange(tensor_dim)) {
        if (order_has_tensor_name.test(tensor_idx)) {
          continue;
        }
        setNewSizesStridesNamesFor(ellipsis_idx, tensor_idx);
        ellipsis_idx++;
      }

      check_names_valid_for(out_dim, new_names);

      Tensor result;
      {
        NoNamesGuard guard;
        result = tensor.as_strided(new_sizes, new_strides);
      }
      internal_set_names_inplace(result, move(new_names), /*validate_names=*/false);
      return result;
        */
}

pub fn align_to_b(
    tensor: &Tensor,
    names:  DimnameList) -> Tensor {

    todo!();
        /*
            auto tensor_names = tensor.names();
      auto tensor_sizes = tensor.sizes();
      auto tensor_strides = tensor.strides();
      vector<i64> new_sizes(names.size(), 1);
      vector<i64> new_strides(names.size(), 0);

      for (const auto idx : irange(tensor_names.size())) {
        const auto& dim = tensor_names[idx];
        TORCH_CHECK(dim.isBasic(),
            "align_to: All input dims must be named. Found unnamed dim at index ",
            idx, " of Tensor", tensor_names);
        auto it = find(names.begin(), names.end(), dim);
        TORCH_CHECK(it != names.end(),
            "align_to: Cannot find dim ", dim, " from Tensor", names,
            " in desired alignment ", names, ".");
        i64 new_idx = distance(names.begin(), it);
        new_sizes[new_idx] = tensor_sizes[idx];
        new_strides[new_idx] = tensor_strides[idx];
      }
      Tensor result;
      {
        NoNamesGuard guard;
        result = tensor.as_strided(new_sizes, new_strides);
      }
      internal_set_names_inplace(result, names);
      return result;
        */
}

pub fn align_as(
        tensor: &Tensor,
        other:  &Tensor) -> Tensor {
    
    todo!();
        /*
            return native::align_to(tensor, other.names());
        */
}

pub fn align_tensors_to(
        tensors: TensorList,
        names:   DimnameList) -> Vec<Tensor> {
    
    todo!();
        /*
            vector<Tensor> result;
      result.reserve(tensors.size());
      for (const auto& tensor : tensors) {
        result.emplace_back(align(tensor, names, /*is_aligning_two_tensors=*/true));
      }
      return result;
        */
}

pub fn align_tensors(tensors: TensorList) -> Vec<Tensor> {
    
    todo!();
        /*
            auto longest_dim = max_element(
          tensors.begin(), tensors.end(),
          [](const Tensor& a, const Tensor& b) {
            return a.dim() < b.dim();
          });
      return align_tensors_to(tensors, longest_dim->names());
        */
}

/**
  | Misc. Dimname overloads that don't
  | have homes. Maybe we should move all
  | of them here or autogenerate them because
  | they look so similar.
  |
  */
pub fn gather(
        self_:       &Tensor,
        dim:         Dimname,
        index:       &Tensor,
        sparse_grad: bool) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("gather");
        */
}

pub fn gather_out(
        self_:       &Tensor,
        dim:         Dimname,
        index:       &Tensor,
        sparse_grad: bool,
        result:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("gather");
        */
}

pub fn index_add_a(
        self_:  &Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor,
        alpha:  &Scalar) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("index_add");
        */
}

pub fn index_add_b(
        self_:  &mut Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor,
        alpha:  &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("index_add");
        */
}

pub fn index_fill_a(
        self_:  &Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Scalar) -> Tensor {
    
    todo!();
        /*
            return index_fill(self, dimname_to_position(self, dim), index, source);
        */
}

pub fn index_fill_b(
        self_:  &mut Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self.index_fill_(dimname_to_position(self, dim), index, source);
        */
}

pub fn index_fill_c(
        self_:  &Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            return index_fill(self, dimname_to_position(self, dim), index, source);
        */
}

pub fn index_fill_d(
        self_:  &mut Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self.index_fill_(dimname_to_position(self, dim), index, source);
        */
}


pub fn index_copy_a(
        self_:  &Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("index_copy");
        */
}


pub fn index_copy_b(
        self_:  &mut Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("index_copy");
        */
}


pub fn index_select_out(
        self_: &Tensor,
        dim:   Dimname,
        index: &Tensor,
        out:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("index_select");
        */
}


pub fn index_select(
        self_: &Tensor,
        dim:   Dimname,
        index: &Tensor) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("index_select");
        */
}

pub fn scatter_a(
        self_:  &Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("scatter");
        */
}

pub fn scatter_b(
        self_:  &mut Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("scatter");
        */
}


pub fn scatter_c(
        self_:  &Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Scalar) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("scatter");
        */
}


pub fn scatter_d(
        self_:  &mut Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("scatter");
        */
}


pub fn scatter_add_a(
        self_:  &Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("scatter_add");
        */
}


pub fn scatter_add_b(
        self_:  &mut Tensor,
        dim:    Dimname,
        index:  &Tensor,
        source: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("scatter_add");
        */
}


pub fn sort_out_a(
        self_:   &Tensor,
        stable:  Option<bool>,
        dim:     Dimname,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            reportNYIDimnameOverload("sort");
        */
}

pub fn sort_out_b(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool,
        values:  &mut Tensor,
        indices: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            reportNYIDimnameOverload("sort");
        */
}


pub fn sort_a(
        self_:   &Tensor,
        stable:  Option<bool>,
        dim:     Dimname,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            reportNYIDimnameOverload("sort");
        */
}


pub fn sort_b(
        self_:   &Tensor,
        dim:     Dimname,
        keepdim: bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            reportNYIDimnameOverload("sort");
        */
}


pub fn squeeze_a(
        self_: &mut Tensor,
        dim:   Dimname) -> &mut Tensor {
    
    todo!();
        /*
            reportNYIDimnameOverload("squeeze");
        */
}

pub fn squeeze_b(
        self_: &Tensor,
        dim:   Dimname) -> Tensor {
    
    todo!();
        /*
            return squeeze(self, dimname_to_position(self, dim));
        */
}
