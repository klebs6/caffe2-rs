crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Linear.cpp]

pub fn linear(
        input:    &Tensor,
        weight:   &Tensor,
        bias_opt: &Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      auto bias = bias_opt.has_value()
        ? MaybeOwned<Tensor>::borrowed(*bias_opt)
        : MaybeOwned<Tensor>::owned(in_place);

      if (input.is_mkldnn()) {
        return mkldnn_linear(input, weight, *bias);
      }
    #if defined(C10_MOBILE)
      if (xnnpack::use_linear(input, weight, *bias)) {
        return xnnpack::linear(input, weight, *bias);
      }
    #endif
      if (input.dim() == 2 && bias->defined()) {
        // Fused op is marginally faster.
        return addmm(*bias, input, weight.t());
      }
      auto output = matmul(input, weight.t());
      if (bias->defined()) {
        output.add_(*bias);
      }
      return output;
        */
}

/**
  | sumproduct_pair computes
  | `(left*right).sum(sumdims)` by means of
  | permutation and batch matrix multiplication
  |
  | its main purpose is to provide a pairwise
  | reduction for einsum
  |
  */
pub fn sumproduct_pair(
        left:     &Tensor,
        right:    &Tensor,
        sum_dims: &[i32],
        keepdim:  bool) -> Tensor {
    
    todo!();
        /*
            // assumes that tensors have been pre-unsqueezed (so that all dimensions match - after broadcasting)
      // but makes no other assumptions on the order of dimensions
      TORCH_CHECK(left_.dim()==right_.dim(), "number of dimensions must match");
      if (sum_dims_.size() == 0)
        return mul(left_, right_);
      i64 dim = left_.dim();
      auto sum_dims = dim_list_to_bitset(sum_dims_, dim);
      // dimensions that will be part of the output (i.e. not summed over) in three vectors
      // dims in lro appear in left, right and output, similarly lo: left and output, ro: right and output
      // also the sizes are kept track of for reshaping
      vector<i64> lro, lo, ro;
      i64 lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
      Tensor left = left_;
      Tensor right = right_;
      for (const auto i : irange(dim)) {
        auto sl = left.size(i)>1;
        auto sr = right.size(i)>1;
        if (sum_dims[i]) { // first dimensions that will be summed over after multiplication
          if (sl && sr) {  // dimensions nontrivially in both left and right must be of the same size
            TORCH_CHECK(left.size(i)==right.size(i), "non-broadcast dimensions must match");
            sum_size *= left.size(i);
          } else if (sl) { // if it is only in one of left and right, we can sum right away
            left = left.sum(i, true);
          } else if (sr) {
            right = right.sum(i, true);
          }
        } else if (sl && sr) { // now deal with dimensions  dimensions that will be in the output
          // dimensions nontrivially in both left and right must be of the same size
          TORCH_CHECK(left.size(i)==right.size(i), "non-broadcast dimensions must match");
          lro.push_back(i);
          lro_size *= left.size(i);
        } else if (sl) { // keep track of dimensions appearing only once
          lo.push_back(i);
          lo_size *= left.size(i);
        } else {
          ro.push_back(i);
          ro_size *= right.size(i);
        }
      }
      // we now work with the following permutations / shapes.
      // the pipeline is permute inputs -> reshape inputs -> batch matrix mul -> reshape(view) output -> permute output
      // output: "lro, lo, 1-for-summed-dims, ro" with orgiginal shape dimensions
      // left:   "lro, lo, summed" permuted with lpermutation and the three flattened
      // right:  "lro, summed, ro" permuted with rpermutation and the three flattened
      // then the permuted output is a view of bmm(left, right)
      // finally, opermutation reverts the permutation to the original order of dimensions
      vector<i64> out_size;
      for (auto& d : lro) out_size.push_back(left.size(d));
      for (auto& d : lo) out_size.push_back(left.size(d));
      for (auto& d : sum_dims_) { out_size.push_back(1); (void)(d); }; // avoid warining about not using d
      for (auto& d : ro) out_size.push_back(right.size(d));

      vector<i64> lpermutation(lro);
      lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
      lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
      lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

      vector<i64> rpermutation(lro);
      rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
      rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
      rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

      vector<i64> opermutation(lro.size()+lo.size()+sum_dims_.size()+ro.size(), -1);
      {
        i64 i = 0;

        for (auto it = lro.cbegin(); it != lro.cend(); i++, it++) {
          opermutation[*it] = i;
        }
        for (auto it = lo.cbegin(); it != lo.cend(); i++, it++) {
          opermutation[*it] = i;
        }
        for (auto it = sum_dims_.cbegin(); it != sum_dims_.cend(); i++, it++) {
          opermutation[*it] = i;
        }
        for (auto it = ro.cbegin(); it != ro.cend(); i++, it++) {
          opermutation[*it] = i;
        }
      }

      // now we can execute the operations above
      left = left.permute(lpermutation).reshape({lro_size, lo_size, sum_size});
      right = right.permute(rpermutation).reshape({lro_size, sum_size, ro_size});
      Tensor result = bmm(left, right);
      result = result.view(out_size).permute(opermutation);

      // finally squeeze summed dimensions if desired
      if (! keepdim) {
        auto sizes = result.sizes().vec();
        for (int i = dim-1; i>=0; i--) {
          if (sum_dims[i]) {
            sizes.erase(sizes.begin() + i);
          }
        }
        result = result.view(sizes);
      }
      return result;
        */
}

pub fn einsum_check_label(label: u8) -> bool {
    
    todo!();
        /*
            return isalpha(label);
        */
}

pub fn einsum_label_to_index(label: u8) -> u8 {
    
    todo!();
        /*
            constexpr u8 NUM_OF_LETTERS = 'z' - 'a' + 1;
      return isupper(label) ? label - 'A' : NUM_OF_LETTERS + (label - 'a');
        */
}

pub fn einsum_index_to_label(index: u8) -> u8 {
    
    todo!();
        /*
            constexpr u8 NUM_OF_LETTERS = 'z' - 'a' + 1;
      return index < NUM_OF_LETTERS ? index + 'A' : index - NUM_OF_LETTERS + 'a';
        */
}

/**
  | There are roughly three parts to compute
  | einsum:
  |
  | 1. Parse equation to extract the labels for
  | each input operand and output
  |
  | 2. Unsqueeze missing dimensions from input
  | operands and permute to align them
  |
  | 3. Compute result by multiplying input operands
  |    and summing contraction dimensions We do the
  |    last part by reducing to bmm.
  |
  */
pub fn einsum(
        equation: StringView,
        operands: &[Tensor]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!operands.empty(), "einsum(): must provide at least one operand");

      // Code used to identify ELLIPSIS ("...")
      constexpr u8 ELLIPSIS = 52;

      // Find arrow (->) to split equation into lhs and rhs
      const auto arrow_pos = equation.find("->");
      const auto lhs = equation.substr(0, arrow_pos);

      const auto num_ops = operands.size();

      // Convert labels for input operands into an index in [0, 52) and store
      // them in op_labels for each operand along with ELLIPSIS if present.
      vector<vector<u8>> op_labels(num_ops);
      bool found_ell = false;
      usize curr_op = 0;
      for (auto i = decltype(lhs.length()){0}; i < lhs.length(); ++i) {
        const unsigned char label = lhs[i];
        switch (label) {
          case ' ':
            // Ignore spaces
            break;

          case '.':
            TORCH_CHECK(
                // Only one ellipsis per operand can be given
                !found_ell,
                "einsum(): found \'.\' for operand ",
                curr_op,
                " for which an ellipsis was already found");
            TORCH_CHECK(
                // Ensure it's a valid ellipsis
                i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.',
                "einsum(): found \'.\' for operand ",
                curr_op,
                " that is not part of any ellipsis");
            op_labels[curr_op].push_back(ELLIPSIS);
            found_ell = true;
            break;

          case ',':
            // Move onto next operand
            ++curr_op;
            TORCH_CHECK(
                curr_op < num_ops,
                "einsum(): fewer operands were provided than specified in the equation");
            found_ell = false;
            break;

          default:
            // Parse label
            TORCH_CHECK(
                einsum_check_label(label),
                "einsum(): invalid subscript given at index ",
                i,
                " in the equation string, subscripts must be in [a-zA-Z]");
            op_labels[curr_op].push_back(einsum_label_to_index(label));
        }
      }

      TORCH_CHECK(
          curr_op == num_ops - 1,
          "einsum(): more operands were provided than specified in the equation");

      // Labels must be within [a-zA-Z].
      constexpr u8 TOTAL_LABELS = 52;
      vector<i64> label_count(TOTAL_LABELS, 0);

      // The maximum number of dimensions covered by any ellipsis, needed when
      // unsqueezing missing dimensions from operands to permute and broadcast
      i64 ell_num_dim = 0;

      // Compute label frequency and number of dimensions covered by ellipsis
      // We do this after parsing labels to make it more readable and simpler
      // to compute the number of dimensions covered by ellipsis.
      for(const auto i : irange(num_ops)) {
        const auto operand = operands[i];
        const auto labels = op_labels[i];
        const auto ndims = operand.dim();
        i64 nlabels = static_cast<i64>(labels.size());
        bool has_ellipsis = false;

        for (const auto& label : labels) {
          if (label == ELLIPSIS) {
            --nlabels;
            has_ellipsis = true;
            ell_num_dim = max(ell_num_dim, ndims - nlabels);
          } else {
            ++label_count[label];
          }
        }

        TORCH_CHECK(
            has_ellipsis ? nlabels <= ndims : nlabels == ndims,
            "einsum(): the number of subscripts in the equation (",
            nlabels,
            has_ellipsis ? ") is more than the number of dimensions ("
                         : ") does not match the number of dimensions (",
            ndims,
            ") for operand ",
            i,
            has_ellipsis ? "" : " and no ellipsis was given");
      }

      // We want to align the dimensions of every input tensor to have
      // shape out_dims + sum_dims. For this, we create a mapping of label
      // to index into the permuted shape.
      vector<i64> label_perm_index(TOTAL_LABELS, -1);

      // Current index in the permuted shape
      i64 perm_index = 0;

      // Start index of ellipsis dimensions in the permuted shape
      i64 ell_index = 0;
      found_ell = false;

      if (arrow_pos == string::npos) {
        // Implicit output is ellipsis (...) + labels seen only once
        perm_index = ell_num_dim;
        found_ell = true;
        for (const auto label : irange(TOTAL_LABELS)) {
          if (label_count[label] == 1) {
            label_perm_index[label] = perm_index++;
          }
        }
      } else {
        // Parse explicit output
        const auto rhs = equation.substr(arrow_pos + 2);
        for (auto i = decltype(rhs.length()){0}; i < rhs.length(); ++i) {
          const unsigned char label = rhs[i];
          switch (label) {
            case ' ':
              // Ignore spaces
              break;

            case '.':
              TORCH_CHECK(
                  // There can only be one ellipsis in the output
                  !found_ell,
                  "einsum(): found \'.\' for output but an ellipsis (...) was already found");
              TORCH_CHECK(
                  // Ensure ellipsis is correct
                  i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.',
                  "einsum(): found \'.\' for output that is not part of any ellipsis (...)");
              ell_index = perm_index;
              perm_index += ell_num_dim;
              found_ell = true;
              break;

            default:
              TORCH_CHECK(
                  einsum_check_label(label),
                  "einsum(): invalid subscript given at index ",
                lhs.size() + 2 + i,
                  " in the equation string, subscripts must be in [a-zA-Z]");
              const auto index = einsum_label_to_index(label);
              TORCH_CHECK(
                  // Ensure label appeared at least once for some input operand and at
                  // most once for the output
                  label_count[index] > 0 && label_perm_index[index] == -1,
                  "einsum(): output subscript ",
                  label,
                  label_perm_index[index] > -1
                      ? " appears more than once in the output"
                      : " does not appear in the equation for any input operand");
              label_perm_index[index] = perm_index++;
          }
        }
      }

      // Save output size before adding contraction dims (dims to sum out)
      const i64 out_size = perm_index;

      // If ellipsis is not part of the output, add to contraction dimensions
      if (!found_ell) {
        ell_index = perm_index;
        perm_index += ell_num_dim;
      }

      // Add contraction labels (labels not present in output)
      for (const auto label : irange(TOTAL_LABELS)) {
        if (label_count[label] > 0 && label_perm_index[label] == -1) {
          label_perm_index[label] = perm_index++;
        }
      }

      // Here we unsqueeze missing dimensions to make all operands have the same
      // number of dimensions. We take diagonals for repeated labels within the
      // same operand. Finally we permute the operands to align dimensions as
      // per the perm_out_index we computed above.
      vector<Tensor> permuted_operands;
      for (const auto i: irange(num_ops)) {
        vector<i64> perm_shape(perm_index, -1);
        vector<i64> label_dim(TOTAL_LABELS, -1);
        Tensor operand = operands[i];
        const auto labels = op_labels[i];
        const auto original_sizes = operand.sizes();

        i64 j = 0;
        for (const auto& label : labels) {
          if (label == ELLIPSIS) {
            // Add missing dimensions covered by the ellipsis
            const auto num_missing_dim =
                ell_num_dim - (original_sizes.size() - labels.size() + 1);
            for (const auto k : irange(num_missing_dim)) {
              (void)k; //Suppress unused warning
              operand = operand.unsqueeze(j);
            }
            for (const auto k : irange(ell_num_dim)) {
              perm_shape[ell_index + k] = j++;
            }
          } else if (label_dim[label] != -1) {
            // Repeated label, take diagonal
            const auto dim = label_dim[label];
            TORCH_CHECK(
                operand.size(j) == operand.size(dim),
                "einsum(): subscript ",
                einsum_index_to_label(label),
                " is repeated for operand ",
                i,
                " but the sizes don't match, ",
                operand.size(j),
                " != ",
                operand.size(dim));
            operand = operand.diagonal(0, dim, j).movedim(-1, dim);
          } else {
            // Lookup output index for label
            label_dim[label] = j;
            perm_shape[label_perm_index[label]] = j++;
          }
        }

        // Add dimensions for missing labels
        for (i64& index : perm_shape) {
          if (index == -1) {
            operand = operand.unsqueeze(-1);
            index = j++;
          }
        }

        permuted_operands.push_back(operand.permute(perm_shape));
      }

      // Check if operands broadcast and keep track of last operand with
      // dimension size != 1 for optimizing reductions
      vector<usize> dim_last_op(perm_index, 0);
      bool has_zero_size_dim = false;
      for (const auto dim : irange(perm_index)) {
        auto broadcast_size = permuted_operands[0].size(dim);
        for (const auto i: irange(1, num_ops)) {
          const auto dim_size = permuted_operands[i].size(dim);
          if (broadcast_size != dim_size && broadcast_size != 1 && dim_size != 1) {
            ostringstream msg;
            msg << "einsum(): operands do not broadcast with remapped shapes [original->remapped]:";
            for (const auto j: irange(num_ops)) {
              msg << " " << operands[j].sizes() << "->"
                  << permuted_operands[j].sizes();
            }
            TORCH_CHECK(false, msg.str());
          }
          if (dim_size != 1) {
            broadcast_size = dim_size;
            dim_last_op[dim] = i;
          }
        }
        has_zero_size_dim |= broadcast_size == 0;
      }

      // Compute result
      Tensor result = permuted_operands[0];

      // Fast path for when an operand has zero sized dim
      if (has_zero_size_dim) {
        vector<i64> out_shape(out_size);
        for (const auto i : irange(out_size)) {
          out_shape[i] = permuted_operands[dim_last_op[i]].size(i);
        }
        return zeros(out_shape, result.options());
      }

      // Sum out or squeeze dimensions that are size 1 for all later operands
      i64 dim = out_size;
      for (i64 i = dim; i < perm_index; ++i, ++dim) {
        if (dim_last_op[i] == 0) {
          if (result.size(dim) == 1) {
            result = result.squeeze(dim--);
          } else {
            result = result.sum(dim--);
          }
        }
      }

      for (const auto i: irange(1, num_ops)) {
        Tensor operand = permuted_operands[i];
        vector<i64> sum_dims;

        // Sum out or squeeze dimensions that are size 1 for all later operands
        dim = out_size;
        for (i64 j = dim; j < perm_index; ++j, ++dim) {
          if (dim_last_op[j] < i) {
            operand = operand.squeeze(dim);
            --dim;
          } else if (dim_last_op[j] == i) {
            if (result.size(dim) == 1) {
              operand = operand.sum(dim);
              result = result.squeeze(dim);
              --dim;
            } else {
              sum_dims.push_back(dim);
            }
          }
        }

        // Multiply tensors and sum out dimensions in sum_dims
        if (sum_dims.empty()) {
          result = result.mul(operand);
        } else if (sum_dims.size() == result.sizes().size()) {
          result = result.flatten().dot(operand.flatten());
        } else {
          result = sumproduct_pair(result, operand, sum_dims, false);
        }
      }

      return result;
        */
}

/**
  | _trilinear computes a trilinear einstein sum
  | with an unrolled dimension
  |
  | the result is
  | `(i1.unsqueeze(expand1)*i2.unsqueeze(expand2)*i2.unsqueeze(expand3)).sum(sumdim)`
  |
  | the computation is unrolled in the unroll_dim
  | dimension
  |
  | its main purpose is to unify the computations
  | in bilinear and bilinear_backward
  |
  */
pub fn trilinear(
    i1:         &Tensor,
    i2:         &Tensor,
    i3:         &Tensor,
    expand1:    &[i32],
    expand2:    &[i32],
    expand3:    &[i32],
    sumdim:     &[i32],
    unroll_dim: i64) -> Tensor {
    
    todo!();
        /*
            i64 total_dim = i1_.dim()+expand1_.size();
      TORCH_CHECK((unroll_dim >= 0) && (unroll_dim < total_dim), "unroll_dim must be in [0,", total_dim-1, "]");
      auto expand1 = dim_list_to_bitset(expand1_, total_dim);
      auto expand2 = dim_list_to_bitset(expand2_, total_dim);
      auto expand3 = dim_list_to_bitset(expand3_, total_dim);
      auto sumdim  = dim_list_to_bitset(sumdim_,  total_dim);
      Tensor i1 = i1_;
      Tensor i2 = i2_;
      Tensor i3 = i3_;
      vector<i64> output_size;
      vector<i64> sum_dims_12, sum_dims_23;
      i64 unroll_size = -1;
      // asserts...
      for (const auto i : irange(total_dim)) {
        i64 s = 0;
        if (expand1[i]) {
          i1 = i1.unsqueeze(i);
        } else  {
          s = i1.size(i);
        }
        if (expand2[i]) {
          i2 = i2.unsqueeze(i);
        } else  {
          s = i2.size(i);
        }
        if (expand3[i]) {
          i3 = i3.unsqueeze(i);
          if (sumdim[i] && (i != unroll_dim))
            sum_dims_12.push_back(i);
        } else  {
          s = i3.size(i);
          if (sumdim[i] && (i != unroll_dim))
            sum_dims_23.push_back(i);
        }
        output_size.push_back(sumdim[i] ? 1 : s);
        if (i == unroll_dim)
          unroll_size = s;
      }
      i64 slicemul1 = (expand1[unroll_dim] ? 0 : 1);
      i64 slicemul2 = (expand2[unroll_dim] ? 0 : 1);
      i64 slicemul3 = (expand3[unroll_dim] ? 0 : 1);

      auto output = zeros(output_size, i1.options());
      if (! sumdim[unroll_dim]) {
        for (const auto k : irange(unroll_size)) {
          Tensor buf = native::sumproduct_pair(i1.narrow(unroll_dim, k * slicemul1, 1),
                                                   i2.narrow(unroll_dim, k * slicemul2, 1),
                                                   sum_dims_12, true);
          buf = native::sumproduct_pair(buf, i3.narrow(unroll_dim, k * slicemul3, 1), sum_dims_23, true);
          output.narrow(unroll_dim, k, 1).add_(buf);
        }
      }
      else {
        for (const auto k : irange(unroll_size)) {
          Tensor buf = native::sumproduct_pair(i1.narrow(unroll_dim, k*slicemul1, 1),
                                                   i2.narrow(unroll_dim, k*slicemul2, 1), sum_dims_12, true);
          buf = native::sumproduct_pair(buf, i3.narrow(unroll_dim, k*slicemul3, 1), sum_dims_23, true);
          output.add_(buf);
        }
      }
      for (i64 i = output.dim()-1; i >= 0; i--)
        if (sumdim[i])
          output.squeeze_(i);
      return output;
        */
}

pub fn bilinear(
        input1:   &Tensor,
        input2:   &Tensor,
        weight:   &Tensor,
        bias_opt: &Option<Tensor>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      TORCH_CHECK(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got ", input1.dim(), " and ", input2.dim());
      for (const auto i : irange(input1.dim() - 1)) {
        TORCH_CHECK(input1.size(i) == input2.size(i),
                  "bilinear(): input batch dimensions do not match at dim ", i, ": got ", input1.size(i), " and ", input2.size(i));
      }
      TORCH_CHECK(input1.size(input1.dim() - 1) == weight.size(1),
                "bilinear(): input1 size does not match weight size: got ",
                input1.size(input1.dim() - 1), " but expected ", weight.size(1));
      TORCH_CHECK(input2.size(input2.dim() - 1) == weight.size(2),
                "bilinear(): input2 size does not match weight size: got ",
                input2.size(input2.dim() - 1), " but expected ", weight.size(2));
      TORCH_CHECK(!bias.defined() || bias.size(0) == weight.size(0),
                "bilinear(): bias size does not match weight size: got ",
                bias.size(0), " but expected ", weight.size(0));

      vector<i64> output_size;
      auto size1 = input1.sizes();
      output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
      output_size.push_back(weight.size(0));
      auto input1_flattened = input1.view({-1, input1.size(-1)});
      auto input2_flattened = input2.view({-1, input2.size(-1)});
      Tensor output = _trilinear(input1_flattened, weight, input2_flattened, {1,3}, {0}, {1,2}, {2,3}).reshape(output_size);
      if (bias.defined()) {
        output = output + bias;
      }
      return output;
        */
}

/**
  | implements tensordot, a matrix-multiplication-like
  | contraction, but the dimensions given
  | in the two dimension lists
  |
  */
pub fn tensordot(
        input1: &Tensor,
        input2: &Tensor,
        dims1:  &[i32],
        dims2:  &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(dims1.size() == dims2.size(), "both dimension lists should have same length");
      i64 csize = 1;  // total size of the contracted dimensions
      Tensor t1 = input1;
      Tensor t2 = input2;
      for (const auto i : irange(dims1.size())) {
        int s1 = input1.size(dims1[i]);
        int s2 = input2.size(dims2[i]);
        if (s2 == 1) { // broadcasted dimensions can be summed right away
          t1 = t1.sum(dims1[i], true);
        } else if (s1 == 1) {
          t2 = t2.sum(dims2[i], true);
        } else {
          TORCH_CHECK(s1 == s2, "contracted dimensions need to match, but first has size ", s1, " in dim ", dims1[i],
                   " and second has size ", s2, " in dim ", dims2[i]);
          csize *= s1;
        }
      }

      auto cdims1 = dim_list_to_bitset(dims1, input1.dim());
      auto cdims2 = dim_list_to_bitset(dims2, input2.dim());
      vector<i64> p1, p2, rsizes;  // p1, p2: input permutations, rsizes: sizes of the result
      p1.reserve(input1.dim());
      p2.reserve(input2.dim());
      rsizes.reserve(input1.dim() + input2.dim() - (i64) dims1.size());
      i64 size1 = 1; // number of non-contracted elements in input1
      i64 size2 = 1; // number of non-contracted elements in input2

      // fill the permutations and compute sizes
      for (const auto i : irange(input1.dim())) {
        if (! cdims1[i]) {
          p1.emplace_back(i);
          size1 *= t1.size(i);
          rsizes.emplace_back(t1.size(i));
        }
      }
      for (const auto i : irange(dims1.size())) {
        p1.emplace_back(dims1[i]);
      }
      for (const auto i : irange(dims2.size())) {
        p2.emplace_back(dims2[i]);
      }
      for (const auto i : irange(input2.dim())) {
        if (! cdims2[i]) {
          p2.emplace_back(i);
          size2 *= t2.size(i);
          rsizes.emplace_back(t2.size(i));
        }
      }
      // permut and reshape for matrix multiplication
      t1 = t1.permute(p1).reshape({size1, csize});
      t2 = t2.permute(p2).reshape({csize, size2});
      // multiply and reshape to target size
      return mm(t1, t2).reshape(rsizes);
        */
}

pub fn tensordot_out<'a>(
        input1: &Tensor,
        input2: &Tensor,
        dims1:  &[i32],
        dims2:  &[i32],
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            Tensor result_tmp = native::tensordot(input1, input2, dims1, dims2);
      auto result_dtype = result_tmp.scalar_type();
      auto output_tensor_dtype = result.scalar_type();
      auto output_device = result.device();
      auto input1_device = input1.device();
      auto input2_device = input2.device();
      // check if the input & output tensors are on the same device.
      TORCH_CHECK(
        (output_device == input1_device) && (input1_device == input2_device),
        "tensordot: Expected the output and input tensors to be on the "
        "same device, but got the output tensor on ", output_device,
        ", input tensor a on ", input1_device, ", and input tensor b on ", input2_device);
      // check if the computed result has the same dtype as the out tensor
      // (because tensordot does not support type promotion)
      TORCH_CHECK(
        result_dtype == output_tensor_dtype, "tensordot",
        ": Expected the output tensor to have dtype ", result_dtype,
        ", but got an output tensor with dtype ", output_tensor_dtype);
      native::resize_output(result, result_tmp.sizes());
      result.copy_(result_tmp);
      return result;
        */
}
