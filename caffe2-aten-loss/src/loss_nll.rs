crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LossNLL.cpp]

/**
  | Returns a contiguous tensor if the source
  | tensor is defined. Otherwise returns
  | the undefined source tensor unmodified.
  |
  */
#[inline] pub fn optional_contiguous(source: &Tensor) -> Tensor {
    
    todo!();
        /*
            return source.defined() ? source.contiguous() : source;
        */
}

/**
  | Returns the address of the first element
  | of a tensor or nullptr if the tensor is
  | undefined.
  |
  */
#[inline] pub fn optional_data<Scalar>(source: &Tensor) -> *mut Scalar {

    todo!();
        /*
            return source.defined() ? source.data_ptr<Scalar>() : nullptr;
        */
}

pub fn nll_loss_out_frame<Scalar>(
        output:       &mut Tensor,
        total_weight: &mut Tensor,
        input:        &Tensor,
        target:       &Tensor,
        weight:       &Tensor,
        reduction:    i64,
        ignore_index: i64)  {

    todo!();
        /*
            const auto n_dims = input.dim();
      const auto n_classes = input.size(-1);

      Scalar* total_weight_data = total_weight.data_ptr<Scalar>();
      *total_weight_data = 0;

      auto weight_contiguous = optional_contiguous(weight);
      const Scalar* weight_data = optional_data<Scalar>(weight_contiguous);

      if (reduction == Reduction::None && n_dims == 2) {
        const auto batch_size = input.size(0);
        output.resize_({batch_size});

        auto input_acc = input.accessor<Scalar, 2>();
        auto target_acc = target.accessor<i64, 1>();
        auto output_acc = output.accessor<Scalar, 1>();

        parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
          for (auto i = start; i < end; i++) {
            const auto cur_target = target_acc[i];

            if (cur_target == ignore_index) {
              output_acc[i] = 0;
              continue;
            }

            TORCH_CHECK_INDEX(
                cur_target >= 0 && cur_target < n_classes,
                "Target ",
                cur_target,
                " is out of bounds.");

            Scalar cur_weight = weight_data != nullptr ? weight_data[cur_target]
                                                         : static_cast<Scalar>(1);
            output_acc[i] = -input_acc[i][cur_target] * cur_weight;
          }
        });

        return;
      }

      // produce scalar output when reducing or input is 1d
      output.resize_({});

      auto input_contiguous = input.contiguous();
      auto target_contiguous = target.contiguous();

      const Scalar* input_data = input_contiguous.data_ptr<Scalar>();
      const i64* target_data = target_contiguous.data_ptr<i64>();

      const i64 ndim = input.dim();
      TORCH_CHECK(ndim <= 2);
      const i64 batch_size = ndim == 1 ? 1 : input.size(0);
      TORCH_CHECK(target.size(0) == batch_size);

      constexpr i64 cascade_sum_num_levels = 8;
      const i64 level_power =
          max(i64(4), utils::CeilLog2(batch_size) / cascade_sum_num_levels);
      const i64 level_step = (1 << level_power);
      const i64 level_mask = level_step - 1;

      i64 num_ignored = 0;

      Scalar weight_partial_sums[cascade_sum_num_levels] = {0};
      Scalar loss_partial_sums[cascade_sum_num_levels] = {0};
      for (i64 b = 0; b < batch_size; b++) {
        const i64 cur_target = target_data[b];
        if (cur_target == ignore_index) {
          ++num_ignored;
          continue;
        }

        TORCH_CHECK_INDEX(
            cur_target >= 0 && cur_target < n_classes,
            "Target ",
            cur_target,
            " is out of bounds.");

        const auto data = input_data[b * n_classes + cur_target];
        if (weight_data) {
          const Scalar weight_val = weight_data[cur_target];
          loss_partial_sums[0] -= data * weight_val;
          weight_partial_sums[0] += weight_val;
        } else {
          loss_partial_sums[0] -= data;
        }

        for (i64 j = 0; j + 1 < cascade_sum_num_levels; ++j) {
          const auto mask = (level_mask << (j * level_power));
          if (C10_LIKELY((b & mask) != 0)) {
            break;
          }

          weight_partial_sums[j + 1] += weight_partial_sums[j];
          loss_partial_sums[j + 1] += loss_partial_sums[j];

          weight_partial_sums[j] = 0;
          loss_partial_sums[j] = 0;
        }
      }

      const Scalar total_weight_val = !weight_data ?
        static_cast<Scalar>(batch_size - num_ignored) :
        accumulate(begin(weight_partial_sums),
                        end(weight_partial_sums),
                        Scalar{0});

      Scalar output_val = accumulate(begin(loss_partial_sums),
                                            end(loss_partial_sums),
                                            Scalar{0});

      if (reduction == Reduction::Mean &&
          (total_weight_val != 0 || input.numel() == 0)) {
        // allow NaN result for total_weight_val == 0 case, see #15870
        output_val /= total_weight_val;
      }

      // write result to output tensors
      *output.data_ptr<Scalar>() = output_val;
      *total_weight_data = total_weight_val;
        */
}


pub fn nll_loss_forward_out_cpu_template(
        output:       &mut Tensor,
        total_weight: &mut Tensor,
        input:        &Tensor,
        target:       &Tensor,
        weight:       &Tensor,
        reduction:    i64,
        ignore_index: i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
          input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
      TORCH_CHECK(
          target.dim() == 1,
          "1D target tensor expected, multi-target not supported");
      TORCH_CHECK(
          input.size(0) == target.size(0),
          "size mismatch (got input: ",
          input.sizes(),
          ", target: ",
          target.sizes(),
          ")")

      const auto n_classes = input.size(-1);

      TORCH_CHECK(
          !weight.defined() || weight.numel() == n_classes,
          "weight tensor should be defined either for all ",
          n_classes,
          " classes or no classes"
          " but got weight tensor of shape: ",
          weight.sizes());

      total_weight.resize_({});

      AT_DISPATCH_FLOATING_TYPES_AND(
          ScalarType::BFloat16, input.scalar_type(), "nll_loss_out_frame", [&] {
            nll_loss_out_frame<Scalar>(
                output,
                total_weight,
                input,
                target,
                weight,
                reduction,
                ignore_index);
          });
        */
}



pub fn nll_loss_backward_out_frame<Scalar>(
        grad_input:   &mut Tensor,
        grad_output:  &Tensor,
        input:        &Tensor,
        target:       &Tensor,
        weight:       &Tensor,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor)  {

    todo!();
        /*
            const auto n_dims = input.dim();
      const auto n_classes = input.size(-1);

      auto target_acc = target.accessor<i64, 1>();

      auto weight_contiguous = optional_contiguous(weight);
      const Scalar* weight_data = optional_data<Scalar>(weight_contiguous);

      if (reduction == Reduction::None && n_dims == 2) {
        const auto batch_size = input.size(0);
        check_dim_size(grad_output, 1, 0, batch_size);
        auto grad_input_acc = grad_input.accessor<Scalar, 2>();
        auto grad_output_acc = grad_output.accessor<Scalar, 1>();
        parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
          for (auto i = start; i < end; i++) {
            auto cur_target = target_acc[i];
            if (cur_target == ignore_index) {
              continue;
            }
            const Scalar w =
                weight_data ? weight_data[cur_target] : static_cast<Scalar>(1);
            grad_input_acc[i][cur_target] = -w * grad_output_acc[i];
          }
        });
        return;
      }

      const Scalar total_weight_value = *total_weight.data_ptr<Scalar>();
      if (total_weight_value <= 0) {
        return;
      }

      TORCH_CHECK(
          grad_output.dim() <= 1 && grad_output.numel() == 1,
          "Expected a single element grad_output tensor, but got: ",
          grad_output.sizes());
      const Scalar grad_output_value = *grad_output.data_ptr<Scalar>();

      if (input.dim() == 1) {
        auto grad_input_acc = grad_input.accessor<Scalar, 1>();

        const auto cur_target = target_acc[0];
        if (cur_target != ignore_index) {
          TORCH_CHECK_INDEX(
              cur_target >= 0 && cur_target < n_classes,
              "Target ",
              cur_target,
              " is out of bounds.");

          grad_input_acc[cur_target] =
              (reduction != Reduction::Mean && weight_data != nullptr)
              ? -weight_data[cur_target]
              : static_cast<Scalar>(-1);
          grad_input_acc[cur_target] *= grad_output_value;
        }
      } else if (input.dim() == 2) {
        auto grad_input_acc = grad_input.accessor<Scalar, 2>();

        const auto batch_size = input.size(0);
        TORCH_CHECK(target.size(0) == batch_size);

        for (i64 i = 0; i < batch_size; i++) {
          const auto cur_target = target_acc[i];

          if (cur_target != ignore_index) {
            TORCH_CHECK_INDEX(
                cur_target >= 0 && cur_target < n_classes,
                "Target ",
                cur_target,
                " is out of bounds.");

            const Scalar w = weight_data != nullptr ? weight_data[cur_target]
                                                      : static_cast<Scalar>(1);
            grad_input_acc[i][cur_target] = -w * grad_output_value;

            if (reduction == Reduction::Mean) {
              grad_input_acc[i][cur_target] /= total_weight_value;
            }
          }
        }
      }
        */
}

pub fn nll_loss_backward_out_cpu_template(
        grad_input:   &mut Tensor,
        grad_output:  &Tensor,
        input:        &Tensor,
        target:       &Tensor,
        weight:       &Tensor,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
      TORCH_CHECK(
          target.dim() == 1,
          "1D target tensor expected, multi-target not supported");
      TORCH_CHECK(
          input.size(0) == target.size(0),
          "size mismatch (got input: ",
          input.sizes(),
          ", target: ",
          target.sizes(),
          ")")
      TORCH_CHECK(
          total_weight.numel() == 1,
          "expected total_weight to be a  single element tensor, got: ",
          total_weight.sizes(),
          " (",
          total_weight.numel(),
          " elements)");

      grad_input.resize_as_(input);
      grad_input.zero_();

      TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
      TORCH_CHECK(
          !weight.defined() || weight.numel() == input.size(-1),
          "weight tensor should be defined either for all or no classes");

      AT_DISPATCH_FLOATING_TYPES_AND(
          ScalarType::BFloat16,
          input.scalar_type(),
          "nll_loss_backward_out_frame",
          [&] {
            nll_loss_backward_out_frame<Scalar>(
                grad_input,
                grad_output,
                input,
                target,
                weight,
                reduction,
                ignore_index,
                total_weight);
          });
        */
}

pub fn nll_loss_forward_out_cpu<'a>(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        output:       &mut Tensor,
        total_weight: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      nll_loss_forward_out_cpu_template(
          output, total_weight, self, target, weight, reduction, ignore_index);
      return tuple<Tensor&, Tensor&>(output, total_weight);
        */
}

pub fn nll_loss_forward_cpu(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      auto output = empty({0}, self.options());
      auto total_weight = empty({0}, self.options());
      native::nll_loss_forward_out_cpu(
          self, target, weight, reduction, ignore_index, output, total_weight);
      return make_tuple(output, total_weight);
        */
}

pub fn nll_loss_backward_out_cpu<'a>(
        grad_output:  &Tensor,
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor,
        grad_input:   &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      nll_loss_backward_out_cpu_template(
          grad_input,
          grad_output,
          self,
          target,
          weight,
          reduction,
          ignore_index,
          total_weight);
      return grad_input;
        */
}

pub fn nll_loss_backward_cpu(
        grad_output:  &Tensor,
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      auto grad_input = zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      native::nll_loss_backward_out_cpu(
          grad_output,
          self,
          target,
          weight,
          reduction,
          ignore_index,
          total_weight,
          grad_input);
      return grad_input;
        */
}

pub fn cross_entropy_loss(
        self_:        &Tensor,
        target:       &Tensor,
        weight:       &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64) -> Tensor {
    
    todo!();
        /*
            return nll_loss_nd(
          log_softmax(
              self, 1, optTypeMetaToScalarType(self.options().dtype_opt())),
          target,
          weight,
          reduction,
          ignore_index);
        */
}

pub fn nll_loss_out<'a>(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        output:       &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      Tensor total_weight = empty({0}, self.options());
      return get<0>(nll_loss_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
        */
}

pub fn nll_loss(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      return get<0>(nll_loss_forward(self, target, weight, reduction, ignore_index));
        */
}

pub fn nll_loss_nd(
        self_:        &Tensor,
        target:       &Tensor,
        weight:       &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64) -> Tensor {
    
    todo!();
        /*
            if (self.dim() < 2) {
        TORCH_CHECK_VALUE(
            false, "Expected 2 or more dimensions (got ", self.dim(), ")");
      }

      if (self.sizes()[0] != target.sizes()[0]) {
        TORCH_CHECK_VALUE(
            false,
            "Expected input batch_size (",
            self.sizes()[0],
            ") to match target batch_size (",
            target.sizes()[0],
            ").");
      }

      Tensor ret;
      Tensor input_ = self;
      Tensor target_ = target;
      if (input_.dim() == 2) {
        ret = nll_loss(input_, target_, weight, reduction, ignore_index);
      } else if (input_.dim() == 4) {
        ret = nll_loss2d(input_, target_, weight, reduction, ignore_index);
      } else {
        // dim == 3 or dim > 4
        auto n = input_.sizes()[0];
        auto c = input_.sizes()[1];
        auto out_size = input_.sizes().slice(2).vec();
        out_size.insert(out_size.begin(), n);
        if (target_.sizes().slice(1) != input_.sizes().slice(2)) {
          TORCH_CHECK(
              false,
              "Expected target size ",
              IntArrayRef(out_size),
              ", got ",
              target_.sizes());
        }
        input_ = input_.contiguous();
        target_ = target_.contiguous();
        // support empty batches, see #15870
        if (input_.numel() > 0) {
          input_ = input_.view({n, c, 1, -1});
        } else {
          input_ = input_.view({n, c, 0, 0});
        }
        if (target_.numel() > 0) {
          target_ = target_.view({n, 1, -1});
        } else {
          target_ = target_.view({n, 0, 0});
        }
        if (!(reduction == Reduction::None)) {
          ret = nll_loss2d(input_, target_, weight, reduction, ignore_index);
        } else {
          auto out =
              nll_loss2d(input_, target_, weight, reduction, ignore_index);
          ret = out.view(out_size);
        }
      }
      return ret;
        */
}
