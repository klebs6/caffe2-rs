crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LossNLL2d.cpp]

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

#[inline] pub fn check_inputs_nll_loss2d(
        input:  &Tensor,
        target: &Tensor,
        weight: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          target.dim() == 3,
          "only batches of spatial targets supported (3D tensors)"
          " but got targets of dimension: ",
          target.dim());
      TORCH_CHECK(
          input.dim() == 4,
          "only batches of spatial inputs supported (4D tensors), "
          "but got input of dimension: ",
          input.dim());
      TORCH_CHECK(
          !weight.defined() || weight.numel() == input.size(1),
          "weight tensor should be defined either for all or no classes");

      const i64 input0 = input.size(0);
      const i64 input2 = input.size(2);
      const i64 input3 = input.size(3);
      const i64 target0 = target.size(0);
      const i64 target1 = target.size(1);
      const i64 target2 = target.size(2);
      TORCH_CHECK(
          input0 == target0 && input2 == target1 && input3 == target2,
          "size mismatch (got input: ",
          input.sizes(),
          " , target: ",
          target.sizes());
        */
}

#[inline] pub fn check_gradout_shape_nll_loss2d(
        grad_output: &Tensor,
        target:      &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          grad_output.dim() == 3,
          "grad_output must have same dimension as target (3) but got dimension: ",
          grad_output.sizes());

      const i64 grad_output0 = grad_output.size(0);
      const i64 grad_output1 = grad_output.size(1);
      const i64 grad_output2 = grad_output.size(2);
      const i64 target0 = target.size(0);
      const i64 target1 = target.size(1);
      const i64 target2 = target.size(2);
      TORCH_CHECK(
          grad_output0 == target0 && grad_output1 == target1 &&
              grad_output2 == target2,
          "size mismatch (got grad_output: ",
          grad_output.sizes(),
          " target: ",
          target.sizes());
        */
}

pub fn nll_loss2d_forward_out_frame<Scalar>(
        output:       &mut Tensor,
        total_weight: &mut Tensor,
        input:        &Tensor,
        target:       &Tensor,
        weight:       &Tensor,
        reduction:    i64,
        ignore_index: i64)  {

    todo!();
        /*
            const i64 n_classes = input.size(1);

      Scalar* total_weight_data = total_weight.data_ptr<Scalar>();
      *total_weight_data = 0;

      auto weight_contiguous = optional_contiguous(weight);
      const Scalar* weight_data = optional_data<Scalar>(weight_contiguous);

      if (reduction == Reduction::None) {
        const i64 batch_size = input.size(0);
        const i64 H = input.size(2);
        const i64 W = input.size(3);

        output.resize_({batch_size, H, W});
        auto input_acc = input.accessor<Scalar, 4>();
        auto output_acc = output.accessor<Scalar, 3>();
        auto target_acc = target.accessor<i64, 3>();

        parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
          for (i64 b = start; b < end; b++) {
            for (i64 h = 0; h < H; h++) {
              for (i64 w = 0; w < W; w++) {
                const i64 cur_target = (i64)target_acc[b][h][w];

                if (cur_target == ignore_index) {
                  output_acc[b][h][w] = static_cast<Scalar>(0);
                  continue;
                }

                TORCH_CHECK_INDEX(
                    cur_target >= 0 && cur_target < n_classes,
                    "Target ",
                    cur_target,
                    " is out of bounds.");

                // load optional weight value
                const Scalar cur_weight = weight_data != nullptr
                    ? weight_data[cur_target]
                    : static_cast<Scalar>(1);
                output_acc[b][h][w] = -input_acc[b][cur_target][h][w] * cur_weight;
              }
            }
          }
        });

        return;
      }

      // produce scalar outputs for the reduction case
      output.resize_({});

      auto input_contiguous = input.contiguous();
      auto target_contiguous = target.contiguous();

      const Scalar* input_data = input_contiguous.data_ptr<Scalar>();
      const i64* target_data = target_contiguous.data_ptr<i64>();

      const i64 batch_size = input.size(0);
      const i64 map_size = input.size(2) * input.size(3);
      const i64 sample_size = map_size * n_classes;
      const i64 numiter = batch_size * map_size;

      constexpr i64 cascade_sum_num_levels = 8;
      Scalar weight_partial_sums[cascade_sum_num_levels] = {0};
      Scalar loss_partial_sums[cascade_sum_num_levels] = {0};
      const i64 level_power =
          max(i64(4), utils::CeilLog2(numiter) / cascade_sum_num_levels);
      const i64 level_step = (1 << level_power);
      const i64 level_mask = level_step - 1;

      i64 num_ignored = 0;
      for (i64 b = 0; b < batch_size; b++) {
        for (i64 elem = 0; elem < map_size; elem++) {
          const i64 cur_target = target_data[b * map_size + elem];
          if (cur_target == ignore_index) {
            ++num_ignored;
            continue;
          }

          TORCH_CHECK_INDEX(
              cur_target >= 0 && cur_target < n_classes,
              "Target ",
              cur_target,
              " is out of bounds.");

          const auto data = input_data[b * sample_size + cur_target * map_size + elem];
          if (weight_data) {
            const Scalar weight_val = weight_data[cur_target];
            loss_partial_sums[0] -= data * weight_val;
            weight_partial_sums[0] += weight_val;
          } else {
            loss_partial_sums[0] -= data;
          }

          const i64 linear_idx = b * map_size + elem;
          for (i64 j = 0; j + 1 < cascade_sum_num_levels; ++j) {
            const auto mask = (level_mask << (j * level_power));
            if (C10_LIKELY((linear_idx & mask) != 0)) {
              break;
            }

            weight_partial_sums[j + 1] += weight_partial_sums[j];
            loss_partial_sums[j + 1] += loss_partial_sums[j];

            weight_partial_sums[j] = 0;
            loss_partial_sums[j] = 0;
          }
        }
      }

      const Scalar total_weight_val = !weight_data ?
        static_cast<Scalar>(numiter - num_ignored) :
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

      *total_weight_data = total_weight_val;
      *output.data_ptr<Scalar>() = output_val;
        */
}

pub fn nll_loss2d_forward_out_cpu_template(
        output:       &mut Tensor,
        total_weight: &mut Tensor,
        input:        &Tensor,
        target:       &Tensor,
        weight:       &Tensor,
        reduction:    i64,
        ignore_index: i64)  {
    
    todo!();
        /*
            check_inputs_nll_loss2d(input, target, weight);
      total_weight.resize_({});

      AT_DISPATCH_FLOATING_TYPES_AND(
          ScalarType::BFloat16,
          input.scalar_type(),
          "nll_loss2d_forward_out_frame",
          [&] {
            nll_loss2d_forward_out_frame<Scalar>(
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


pub fn nll_loss2d_backward_out_frame<Scalar>(
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
            auto weight_contiguous = optional_contiguous(weight);
      const Scalar* weight_data = optional_data<Scalar>(weight_contiguous);

      if (reduction == Reduction::None) {
        check_gradout_shape_nll_loss2d(grad_output, target);

        const i64 batch_size = input.size(0);
        const i64 H = input.size(2);
        const i64 W = input.size(3);

        auto grad_input_acc = grad_input.accessor<Scalar, 4>();
        auto grad_output_acc = grad_output.accessor<Scalar, 3>();
        auto target_acc = target.accessor<i64, 3>();

        parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
          for (i64 b = start; b < end; b++) {
            for (i64 h = 0; h < H; h++) {
              for (i64 w = 0; w < W; w++) {
                const i64 cur_target = target_acc[b][h][w];
                if (cur_target == ignore_index) {
                  continue;
                }
                const Scalar value =
                    -(weight_data ? weight_data[cur_target]
                                  : static_cast<Scalar>(1));
                const Scalar grad_output_value = grad_output_acc[b][h][w];
                grad_input_acc[b][cur_target][h][w] = value * grad_output_value;
              }
            }
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

      const auto target_contiguous = target.contiguous();
      const i64* target_data = target_contiguous.data_ptr<i64>();

      Scalar* grad_input_data = grad_input.data_ptr<Scalar>();

      const i64 batch_size = input.size(0);
      const i64 n_classes = input.size(1);
      const i64 map_size = input.size(2) * input.size(3);
      const i64 sample_size = map_size * n_classes;

      Scalar normalize = (reduction == Reduction::Mean)
          ? total_weight_value
          : static_cast<Scalar>(1);

      parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
        for (i64 b = start; b < end; b++) {
          for (i64 elem = 0; elem < map_size; elem++) {
            const i64 cur_target = target_data[b * map_size + elem];

            if (cur_target == ignore_index) {
              continue;
            }

            TORCH_CHECK_INDEX(
                cur_target >= 0 && cur_target < n_classes,
                "Target ",
                cur_target,
                " is out of bounds.");

            const i64 index = b * sample_size + cur_target * map_size + elem;
            const Scalar w = weight_data != nullptr ? weight_data[cur_target]
                                                      : static_cast<Scalar>(1);
            grad_input_data[index] = -w / normalize * grad_output_value;
          }
        }
      });
        */
}

pub fn nll_loss2d_backward_out_cpu_template(
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
            check_inputs_nll_loss2d(input, target, weight);
      grad_input.resize_as_(input);
      grad_input.zero_();
      TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
      TORCH_CHECK(
          total_weight.numel() == 1,
          "expected total_weight to be a single element tensor, got: ",
          total_weight.sizes(),
          " (",
          total_weight.numel(),
          " elements)");

      AT_DISPATCH_FLOATING_TYPES_AND(
          ScalarType::BFloat16,
          input.scalar_type(),
          "nll_loss2d_backward_out_frame",
          [&] {
            nll_loss2d_backward_out_frame<Scalar>(
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

pub fn nll_loss2d_forward_out_cpu(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        output:       &mut Tensor,
        total_weight: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      nll_loss2d_forward_out_cpu_template(
          output, total_weight, self, target, weight, reduction, ignore_index);
      return tuple<Tensor&, Tensor&>(output, total_weight);
        */
}

pub fn nll_loss2d_forward_cpu(
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
      native::nll_loss2d_forward_out_cpu(
          self, target, weight, reduction, ignore_index, output, total_weight);
      return make_tuple(output, total_weight);
        */
}

pub fn nll_loss2d_backward_out_cpu(
        grad_output:  &Tensor,
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        total_weight: &Tensor,
        grad_input:   &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      nll_loss2d_backward_out_cpu_template(
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

pub fn nll_loss2d_backward_cpu(
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

      auto grad_input = zeros_like(self);
      native::nll_loss2d_backward_out_cpu(
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

pub fn nll_loss2d_out(
        self_:        &Tensor,
        target:       &Tensor,
        weight_opt:   &Option<Tensor>,
        reduction:    i64,
        ignore_index: i64,
        output:       &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      Tensor total_weight = empty({0}, self.options());
      return get<0>(nll_loss2d_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
        */
}

pub fn nll_loss2d(
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

      return get<0>(nll_loss2d_forward(self, target, weight, reduction, ignore_index));
        */
}
