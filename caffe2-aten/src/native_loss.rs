crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Loss.cpp]

pub const EPSILON: f32 = 1e-12;

#[inline] pub fn apply_loss_reduction(
        unreduced: &Tensor,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            if (reduction == Reduction::Mean) {
          return unreduced.mean();
        } else if (reduction == Reduction::Sum) {
          return unreduced.sum();
        }
        return unreduced;
        */
}

define_dispatch!{smooth_l1_stub}
define_dispatch!{smooth_l1_backward_stub}
define_dispatch!{huber_stub}
define_dispatch!{huber_backward_stub}
define_dispatch!{mse_stub}
define_dispatch!{mse_backward_stub}

pub fn cosine_embedding_loss(
        input1:    &Tensor,
        input2:    &Tensor,
        target:    &Tensor,
        margin:    f64,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          target.dim() == 1,
          "1D target tensor expected, multi-target not supported");

      auto prod_sum = (input1 * input2).sum(1);
      auto mag_square1 = (input1 * input1).sum(1) + EPSILON;
      auto mag_square2 = (input2 * input2).sum(1) + EPSILON;
      auto denom = (mag_square1 * mag_square2).sqrt_();
      auto cos = prod_sum / denom;

      auto zeros = zeros_like(cos, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto pos = 1 - cos;
      auto neg = (cos - margin).clamp_min_(0);
      auto output_pos = where(target == 1, pos, zeros);
      auto output_neg = where(target == -1, neg, zeros);
      auto output = output_pos + output_neg;
      return apply_loss_reduction(output, reduction);
        */
}

pub fn hinge_embedding_loss(
        self_:     &Tensor,
        target:    &Tensor,
        margin:    f64,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            auto zeros = zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto margin_clamp = (margin - self).clamp_min_(0);
      auto output_margin = where(target != 1, margin_clamp, zeros);
      auto output_self = where(target != -1, self, zeros);
      auto output = output_margin + output_self;
      return apply_loss_reduction(output, reduction);
        */
}

pub fn triplet_margin_loss(
        anchor:    &Tensor,
        positive:  &Tensor,
        negative:  &Tensor,
        margin:    f64,
        p:         f64,
        eps:       f64,
        swap:      bool,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            auto dist_pos = pairwise_distance(anchor, positive, p, eps);
      auto dist_neg = pairwise_distance(anchor, negative, p, eps);
      if (swap) {
        auto dist_swap = pairwise_distance(positive, negative, p, eps);
        dist_neg = min(dist_neg, dist_swap);
      }
      auto output = clamp_min(margin + dist_pos - dist_neg, 0);
      return apply_loss_reduction(output, reduction);
        */
}

pub fn margin_ranking_loss(
        input1:    &Tensor,
        input2:    &Tensor,
        target:    &Tensor,
        margin:    f64,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            auto output =  (-target * (input1 - input2) + margin).clamp_min_(0);
      return apply_loss_reduction(output, reduction);
        */
}


pub fn kl_div_log_target(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            auto output = exp(target) * (target - input);
      return apply_loss_reduction(output, reduction);
        */
}

pub fn kl_div_non_log_target(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            auto output_pos = target * (log(target) - input);
      auto zeros = zeros_like(output_pos, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto output = where(target > 0, output_pos, zeros);
      return apply_loss_reduction(output, reduction);
        */
}

pub fn kl_div(
        input:      &Tensor,
        target:     &Tensor,
        reduction:  i64,
        log_target: bool) -> Tensor {
    
    todo!();
        /*
            return log_target ? _kl_div_log_target(input, target, reduction)
                        : _kl_div_non_log_target(input, target, reduction);
        */
}

pub fn kl_div_backward_cpu(
        grad:       &Tensor,
        input:      &Tensor,
        target:     &Tensor,
        reduction:  i64,
        log_target: bool) -> Tensor {
    
    todo!();
        /*
            auto grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto grad_expand = grad.expand_as(input);
      if (!log_target) {
        auto iter = TensorIteratorConfig()
          .add_output(grad_input)
          .add_input(target)
          .add_input(grad_expand)
          .build();
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kl_div_backward_cpu", [&]() {
          cpu_serial_kernel(iter, [](Scalar target_val, Scalar grad_val) -> Scalar{
            return target_val > 0 ? -target_val * grad_val : 0;
          });
        });
      }
      else {
        grad_input = -exp(target) * grad_expand;
      }

      if (reduction == Reduction::Mean) {
        return grad_input / input.numel();
      }
      return grad_input;
        */
}

pub fn binary_cross_entropy_cpu(
        input:      &Tensor,
        target:     &Tensor,
        weight_opt: &Option<Tensor>,
        reduction:  i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        Tensor loss = empty_like(input);
        return native::binary_cross_entropy_out_cpu(
            input, target, weight, reduction, loss);
        */
}

pub fn binary_cross_entropy_out_cpu(
        input:      &Tensor,
        target:     &Tensor,
        weight_opt: &Option<Tensor>,
        reduction:  i64,
        loss:       &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        Tensor loss_squeezed = squeeze(loss);

        auto iter = TensorIteratorConfig()
          .add_output(loss_squeezed)
          .add_owned_input(squeeze(input))
          .add_owned_input(squeeze(target))
          .build();

        AT_DISPATCH_FLOATING_TYPES(loss.scalar_type(), "binary_cross_entropy", [&] {
            native::cpu_kernel(
                iter,
                [] (Scalar input_val, Scalar target_val) {
                    TORCH_CHECK(
                        (input_val >= 0) && (input_val <= 1),
                        "all elements of input should be between 0 and 1"
                    );

                    // Binary cross entropy tensor is defined by the equation:
                    // L = -w (y ln(x) + (1-y) ln(1-x))
                    return (target_val - Scalar(1))
                        * max(Scalar(log(Scalar(1) - input_val)), Scalar(-100))
                        - target_val * max(Scalar(log(input_val)), Scalar(-100));
                }
            );
        });
        if (weight.defined()) {
            loss.mul_(weight);
        }
        if (reduction != Reduction::None) {
            Tensor loss_reduced = apply_loss_reduction(loss, reduction);
            loss.resize_as_(loss_reduced).copy_(loss_reduced);
        }
        return loss;
        */
}

pub fn binary_cross_entropy_backward_cpu(
        grad:       &Tensor,
        input:      &Tensor,
        target:     &Tensor,
        weight_opt: &Option<Tensor>,
        reduction:  i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        Tensor grad_input = empty_like(input);
        return native::binary_cross_entropy_backward_out_cpu(
            grad, input, target, weight, reduction, grad_input);
        */
}


pub fn binary_cross_entropy_backward_out_cpu(
        grad:       &Tensor,
        input:      &Tensor,
        target:     &Tensor,
        weight_opt: &Option<Tensor>,
        reduction:  i64,
        grad_input: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

        Tensor grad_input_squeezed = squeeze(grad_input);

        auto iter = TensorIteratorConfig()
          .add_output(grad_input_squeezed)
          .add_owned_input(squeeze(grad))
          .add_owned_input(squeeze(input))
          .add_owned_input(squeeze(target))
          .build();

        AT_DISPATCH_FLOATING_TYPES(grad_input.scalar_type(), "binary_cross_entropy_backward", [&] {
            native::cpu_kernel(
                iter,
                [] (Scalar grad_val, Scalar input_val, Scalar target_val) {
                    // The gradient is the partial derivative of BCELoss
                    // with respect to x
                    // d(L)/d(x) = -w (y - x) / (x - x^2)
                    return grad_val * (input_val - target_val)
                        / (Scalar(max(
                            (Scalar(1) - input_val) * input_val,
                            Scalar(EPSILON)
                        )));
                }
            );
        });
        if (weight.defined()) {
            grad_input.mul_(weight);
        }
        if (reduction == Reduction::Mean) {
            grad_input.div_(input.numel());
        }
        return grad_input;
        */
}

pub fn binary_cross_entropy_with_logits(
        input:          &Tensor,
        target:         &Tensor,
        weight_opt:     &Option<Tensor>,
        pos_weight_opt: &Option<Tensor>,
        reduction:      i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& pos_weight = value_or_else(pos_weight_opt, [] {return Tensor();});

        Tensor loss;
        auto max_val = (-input).clamp_min_(0);
        if (pos_weight.defined()) {
            // pos_weight need to be broadcasted, thus mul(target) is not inplace.
            auto log_weight = (pos_weight - 1).mul(target).add_(1);
            loss = (1 - target).mul_(input).add_(log_weight.mul_(((-max_val).exp_().add_((-input - max_val).exp_())).log_().add_(max_val)));
        } else {
            loss = (1 - target).mul_(input).add_(max_val).add_((-max_val).exp_().add_((-input -max_val).exp_()).log_());
        }

        if (weight.defined()) {
            loss.mul_(weight);
        }

        return apply_loss_reduction(loss, reduction);
        */
}

pub fn binary_cross_entropy_with_logits_backward(
        grad:           &Tensor,
        input:          &Tensor,
        target:         &Tensor,
        weight_opt:     &Option<Tensor>,
        pos_weight_opt: &Option<Tensor>,
        reduction:      i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& pos_weight = value_or_else(pos_weight_opt, [] {return Tensor();});

        Tensor grad_input;
        if (pos_weight.defined()) {
            // pos_weight need to be broadcasted, thus mul(target) is not inplace.
            auto t = pos_weight.mul(target);
            grad_input = t.add(1).sub_(target).mul_(input.sigmoid()).sub_(t).mul_(grad);
        } else {
            grad_input = (input.sigmoid() - target).mul_(grad);
        }

        if (weight.defined()) {
            grad_input.mul_(weight);
        }

        if (reduction == Reduction::Mean) {
            return grad_input / input.numel();
        }

        return grad_input;
        */
}

pub fn poisson_nll_loss(
        input:     &Tensor,
        target:    &Tensor,
        log_input: bool,
        full:      bool,
        eps:       f64,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            Tensor loss;
        if (log_input) {
            loss = exp(input) - target * input;
        } else {
            loss = input - target * log(input + eps);
        }

        if (full) {
            auto stirling_term = target * log(target) - target + 0.5 * log(2 * pi<double> * target);
            loss += stirling_term.masked_fill(target <= 1, 0);
        }

        return apply_loss_reduction(loss, reduction);
        */
}

pub fn soft_margin_loss_backward_out(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
      auto z = exp(-target * input);
      // inplace version of: grad_input = -norm * target * z / (1. + z) * grad_output;
      mul_out(grad_input, target, z).mul_(-norm);
      z.add_(1);
      grad_input.div_(z).mul_(grad_output);
      return grad_input;
        */
}

pub fn soft_margin_loss_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64) -> Tensor {
    
    todo!();
        /*
            auto grad_input = empty({0}, input.options());
      soft_margin_loss_backward_out(grad_input, grad_output, input, target, reduction);
      return grad_input;
        */
}

pub fn soft_margin_loss_out(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        output:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // compute inplace variant of: output = log(1. + exp(-input * target));
      neg_out(output, input).mul_(target).exp_().add_(1.).log_();
      if (reduction != Reduction::None) {
        auto tmp = apply_loss_reduction(output, reduction);
        output.resize_({});
        output.copy_(tmp);
      }
      return output;
        */
}

pub fn soft_margin_loss(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            auto output = empty({0}, input.options());
      soft_margin_loss_out(output, input, target, reduction);
      return output;
        */
}

pub fn smooth_l1_loss(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        beta:      f64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.")
      if (beta == 0) {
          return native::l1_loss(input, target, reduction);
      }
      Tensor loss;
      auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
      smooth_l1_stub(iter.device_type(), iter, beta);
      return apply_loss_reduction(iter.output(), reduction);
        */
}

pub fn smooth_l1_loss_out(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        beta:      f64,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.")
      if (beta == 0) {
          return native::l1_loss_out(input, target, reduction, result);
      }
      if (reduction != Reduction::None) {
        Tensor loss;
        auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
        smooth_l1_stub(iter.device_type(), iter, beta);
        if (reduction == Reduction::Mean) {
          mean_out(result, iter.output(), 0);
        } else {
          sum_out(result, iter.output(), 0);
        }
      } else {
        auto iter = TensorIterator::borrowing_binary_op(result, input, target);
        smooth_l1_stub(iter.device_type(), iter, beta);
      }
      return result;
        */
}


pub fn smooth_l1_loss_backward_out(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        beta:        f64,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (beta <= 0)
        return native::l1_loss_backward_out(
            grad_output, input, target, reduction, grad_input);
      auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
      auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(input)
        .add_input(target)
        .add_input(grad_output)
        .build();
      smooth_l1_backward_stub(iter.device_type(), iter, norm, beta);
      return grad_input;
        */
}


pub fn smooth_l1_loss_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        beta:        f64) -> Tensor {
    
    todo!();
        /*
            if (beta <= 0)
          return native::l1_loss_backward(grad_output, input, target, reduction);
      auto grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      return smooth_l1_loss_backward_out(grad_input, grad_output, input, target, reduction, beta);
        */
}

pub fn huber_loss(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        delta:     f64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
      Tensor loss = empty_like(input);
      auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
      huber_stub(iter.device_type(), iter, delta);
      return apply_loss_reduction(loss, reduction);
        */
}

pub fn huber_loss_out(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        delta:     f64,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
      auto iter = TensorIterator::borrowing_binary_op(result, input, target);
      huber_stub(iter.device_type(), iter, delta);
      if (reduction != Reduction::None) {
        auto reduced = apply_loss_reduction(result, reduction);
        result.resize_({});
        result.copy_(reduced);
      }
      return result;
        */
}

pub fn huber_loss_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        delta:       f64) -> Tensor {
    
    todo!();
        /*
            auto grad_input = zeros_like(input, MemoryFormat::Contiguous);
      return huber_loss_backward_out(grad_input, grad_output, input, target, reduction, delta);
        */
}

pub fn huber_loss_backward_out(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        delta:       f64,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto norm = (reduction == Reduction::Mean) ? (1. / input.numel()) : 1.;
      auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(input)
        .add_input(target)
        .add_input(grad_output)
        .build();
      huber_backward_stub(iter.device_type(), iter, norm, delta);
      return grad_input;
        */
}

pub fn mse_loss(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            Tensor loss;
      auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
      mse_stub(iter.device_type(), iter);
      return apply_loss_reduction(iter.output(), reduction);
        */
}

pub fn mse_loss_out(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (reduction != Reduction::None) {
        Tensor loss;
        auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
        mse_stub(iter.device_type(), iter);
        if (reduction == Reduction::Mean) {
          mean_out(result, iter.output(), 0);
        } else {
          sum_out(result, iter.output(), 0);
        }
      } else {
        auto iter = TensorIterator::borrowing_binary_op(result, input, target);
        mse_stub(iter.device_type(), iter);
      }
      return result;
        */
}

pub fn mse_loss_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      return mse_loss_backward_out(grad_input, grad_output, input, target, reduction);
        */
}

pub fn mse_loss_backward_out(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;
      auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(input)
        .add_input(target)
        .add_input(grad_output)
        .build();
      mse_backward_stub(iter.device_type(), iter, norm);
      return grad_input;
        */
}

pub fn l1_loss(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            const auto float_type = toValueType(input.scalar_type());
      Tensor result = empty({0}, input.options().dtype(float_type));
      return l1_loss_out(result, input, target, reduction);
        */
}

pub fn l1_loss_out(
        input:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        result:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            if (reduction != Reduction::None) {
        auto diff = sub(input, target);
        auto loss = diff.is_complex() ? diff.abs() : diff.abs_();
        if (reduction == Reduction::Mean) {
          return mean_out(result, loss, IntArrayRef{});
        } else {
          return sum_out(result, loss, IntArrayRef{});
        }
      } else {
        auto diff = input.is_complex() ? sub(input, target) : sub_out(result, input, target);
        return abs_out(result, diff);
      }
        */
}

pub fn l1_loss_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input = zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      return l1_loss_backward_out(grad_input, grad_output, input, target, reduction);
        */
}

pub fn l1_loss_backward_out(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto norm = reduction == Reduction::Mean ? grad_output / input.numel() : grad_output;
      return sub_out(grad_input, input, target).sgn_().mul_(norm);
        */
}
