/*!
  | Copyright (c) 2018 MathInf GmbH, Thomas
  | Viehmann
  | 
  | Licensed under the BSD-3-Clause license
  | 
  | This is the CPU implementation of the
  | Connectionist Temporal Loss.
  | 
  | We mostly follow Graves. 1. Graves et
  | al: http://www.cs.toronto.edu/~graves/icml_2006.pdf
  | 
  | We use the equations from above link,
  | but note that [1] has 1-based indexing
  | and we (of course) use 0-based.
  | 
  | Graves et al call the probabilities
  | y, we use log_probs (also calling them
  | inputs)
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LossCTC.cpp]

/**
  | this ad-hoc converts from targets (l
  | in [1]) to augmented targets (l' in [1])
  | note that no bound-checking is done
  |
  */
#[inline] pub fn get_target_prime<target_t>(
        target: *mut Target,
        offset: i64,
        stride: i64,
        idx:    i64,
        BLANK:  i64) -> i64 {

    todo!();
        /*
            if (idx % 2 == 0) {
        return BLANK;
      } else {
        return target[offset + stride * (idx / 2)];
      }
        */
}

/**
  | This kernel is a relatively straightforward
  | implementation of the alpha calculation in the
  | forward backward algorithm (section 4.1).
  |
  | A (minor) twist is that we are using
  | log-calculations to enhance numerical stability
  | (log_probs and log_alpha).
  |
  | The function returns the loss and the alphas,
  | the alphas are kept for the backward step. The
  | wrapper (ctc_loss below) hides the alphas from
  | the user by only returning the loss.
  |
  */
pub fn ctc_loss_cpu_template<Scalar, const target_scalar_type: ScalarType>(
        log_probs:      &Tensor,
        targets:        &Tensor,
        input_lengths:  &[i32],
        target_lengths: &[i32],
        BLANK:          i64) -> (Tensor,Tensor) {

    todo!();
        /*
            // log_probs: input_len x batch_size x num_labels
      // targets [int64]: batch_size x target_length OR sum(target_lengths)
      constexpr Scalar neginf = -numeric_limits<Scalar>::infinity();
      using target_t = typename conditional<target_scalar_type == kInt, int, i64>::type;

      CheckedFrom c = "ctc_loss_cpu";
      auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
      auto targets_arg = TensorArg(targets, "targets", 2);
      checkScalarType(c, targets_arg, target_scalar_type);
      checkDim(c, log_probs_arg, 3);
      checkDimRange(c, targets_arg, 1, 3);

      i64 batch_size = log_probs.size(1);
      i64 num_labels = log_probs.size(2);
      TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
      TORCH_CHECK((i64) input_lengths.size() == batch_size, "input_lengths must be of size batch_size");
      TORCH_CHECK((i64) target_lengths.size() == batch_size, "target_lengths must be of size batch_size");

      usize tg_target_stride;
      i64 max_target_length = 0;
      vector<i64> tg_batch_offsets(batch_size);
      if (targets.dim() == 1) { // concatenated targets
        i64 pos = 0;
        for (i64 i = 0; i < batch_size; i++) {
          tg_batch_offsets[i] = pos;
          pos += target_lengths[i];
          if (max_target_length < target_lengths[i])
             max_target_length = target_lengths[i];
        }
        tg_target_stride = targets.stride(0);
        checkSize(c, targets_arg, 0, pos);
      }
      else { // batch x max_target_length
        // dim is 2
        i64 tg_batch_stride = targets.stride(0);
        for (i64 i = 0; i < batch_size; i++) {
          tg_batch_offsets[i] = i * tg_batch_stride;
          if (max_target_length < target_lengths[i])
            max_target_length = target_lengths[i];
        }
        tg_target_stride = targets.stride(1);
        checkSize(c, targets_arg, 0, batch_size);
        TORCH_CHECK(targets.size(1) >= max_target_length,
                 "Expected tensor to have size at least ", max_target_length, " at dimension 1, but got size ", targets.size(1), " for ", targets_arg,
                 " (while checking arguments for ", c, ")");
      }
      i64 max_input_length = log_probs.size(0);
      for (i64 b = 0; b < batch_size; b++) {
        TORCH_CHECK(input_lengths[b] <= max_input_length,
                 "Expected input_lengths to have value at most ", max_input_length, ", but got value ", input_lengths[b],
                 " (while checking arguments for ", c, ")");
      }

      Tensor log_alpha = empty({batch_size, log_probs.size(0), 2*max_target_length+1}, log_probs.options());
      Tensor neg_log_likelihood = empty({batch_size}, log_probs.options());

      auto lpp  = log_probs.permute({1,0,2});
      auto log_probs_a_global = lpp.accessor<Scalar, 3>();
      auto log_alpha_a_global = log_alpha.accessor<Scalar, 3>();
      auto targets_data = targets.data_ptr<target_t>();
      auto neg_log_likelihood_a = neg_log_likelihood.accessor<Scalar, 1>();

      // alpha calculation for the first row, the three equations for alpha_1 above eq (6)
      // first the default
      log_alpha.narrow(1, 0, 1).fill_(neginf);
      parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
        for (i64 b = start; b < end; b++) {
          i64 input_length = input_lengths[b];
          i64 target_length = target_lengths[b];
          auto log_probs_a = log_probs_a_global[b];
          auto log_alpha_a = log_alpha_a_global[b];
          i64 tg_batch_offset = tg_batch_offsets[b];

          // the first two items of alpha_t above eq (6)
          log_alpha_a[0][0] = log_probs_a[0][BLANK];
          if (target_length > 0)
            log_alpha_a[0][1] = log_probs_a[0][get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 1, BLANK)];

          // now the loop over the inputs
          for (i64 t=1; t<input_length; t++) {
            for (i64 s=0; s<2*target_length+1; s++) {
              auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
              // this loop over s could be parallel/vectorized, too, but the required items are one index apart
              // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
              // for the cuda implementation, that gave a speed boost.
              // This is eq (6) and (7), la1,2,3 are the three summands. We keep track of the maximum for the logsumexp calculation.

              Scalar la1 = log_alpha_a[t-1][s];
              Scalar lamax = la1;
              Scalar la2, la3;
              if (s > 0) {
                la2 = log_alpha_a[t-1][s-1];
                if (la2 > lamax)
                  lamax = la2;
              } else {
                la2 = neginf;
              }
              if ((s > 1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s-2, BLANK) !=
                              current_target_prime)) {
                la3 = log_alpha_a[t-1][s-2];
                if (la3 > lamax)
                  lamax = la3;
              } else {
                la3 = neginf;
              }
              if (lamax == neginf) // cannot do neginf-neginf
                lamax = 0;
              // this is the assignment of eq (6)
              log_alpha_a[t][s] = log(exp(la1-lamax)+exp(la2-lamax)+exp(la3-lamax))+lamax + log_probs_a[t][current_target_prime];
            }
          }
          // the likelihood is the the sum of the last two alphas, eq (8), the loss is the negative log likelihood
          if (target_length == 0) {
            // if the target is empty then there is no preceding BLANK state and hence there is no path to merge
            neg_log_likelihood_a[b] = -log_alpha_a[input_length-1][0];
          } else {
            Scalar l1 = log_alpha_a[input_length-1][target_length*2];
            Scalar l2 = log_alpha_a[input_length-1][target_length*2-1];
            Scalar m = max(l1, l2);
            m = ((m == neginf) ? 0 : m);
            Scalar log_likelihood = log(exp(l1-m)+exp(l2-m))+m;
            neg_log_likelihood_a[b] = -log_likelihood;
          }
        }
      });

      return make_tuple(neg_log_likelihood, log_alpha);
        */
}

/**
  | This is the backward. It consists of two
  | phases:
  |
  | a) computing the beta analogous to the alphas
  | in the forward (backward half of the
  | forward-backward algorithm) (eq (10) and (11))
  |
  | b) collecting the per-activation characters for
  | all s and wrapping the gradient (eq (16), the
  | collection is the sum)
  |
  */
pub fn ctc_loss_backward_cpu_template<Scalar, const target_scalar_type: ScalarType>(
        grad_out:           &Tensor,
        log_probs:          &Tensor,
        targets:            &Tensor,
        input_lengths:      &[i32],
        target_lengths:     &[i32],
        neg_log_likelihood: &Tensor,
        log_alpha:          &Tensor,
        BLANK:              i64,
        zero_infinity:      bool) -> Tensor {

    todo!();
        /*
            constexpr Scalar neginf = -numeric_limits<Scalar>::infinity();
      using target_t = typename conditional<target_scalar_type == kInt, int, i64>::type;
      i64 max_input_length = log_probs.size(0);
      i64 batch_size = log_probs.size(1);
      i64 num_labels = log_probs.size(2);
      Tensor grad = full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // at this point, this is log of empty sum

      // The admin bits. We don't do much checking and assume that the forward did.
      i64 tg_target_stride;
      i64 max_target_length;
      vector<i64> tg_batch_offsets(batch_size);

      if (targets.dim() == 1) { // concatenated targets
        i64 pos = 0;
        max_target_length = 0;
        for (i64 i = 0; i < batch_size; i++) {
          tg_batch_offsets[i] = pos;
          pos += target_lengths[i];
          if (max_target_length < target_lengths[i])
            max_target_length = target_lengths[i];
        }
        tg_target_stride = targets.stride(0);
      }
      else { // batch x max_target_length
        // dim is 2
        i64 tg_batch_stride = targets.stride(0);
        for (i64 i = 0; i < batch_size; i++) {
          tg_batch_offsets[i] = i * tg_batch_stride;
        }
        tg_target_stride = targets.stride(1);
        max_target_length = targets.size(1);
      }

      Tensor log_beta = empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // could be optimized to use only 2 rows
      auto lpp  = log_probs.permute({1,0,2});
      auto log_probs_a_global = lpp.accessor<Scalar, 3>();
      auto log_alpha_a_global = log_alpha.accessor<Scalar, 3>();
      auto log_beta_a_global = log_beta.accessor<Scalar, 3>();
      auto gp = grad.permute({1,0,2});
      auto grad_a_global = gp.accessor<Scalar, 3>();
      auto targets_data = targets.data_ptr<target_t>();

      parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
        for (i64 b = start; b < end; b++) {
          Scalar nll = neg_log_likelihood.accessor<Scalar, 1>()[b];
          if (zero_infinity &&  nll == numeric_limits<Scalar>::infinity()) {
            grad.narrow(1, b, 1).zero_();
            continue;
          }

          auto log_probs_a = log_probs_a_global[b];
          auto log_alpha_a = log_alpha_a_global[b];
          auto log_beta_a = log_beta_a_global[b];
          auto grad_a = grad_a_global[b];
          i64 input_length = input_lengths[b];
          i64 target_length = target_lengths[b];
          i64 tg_batch_offset = tg_batch_offsets[b];

          // the initialization of beta before eq (10)
          // here we do the fill for each batch item separately, as the input lengths will differ, so the t in which
          // we start varies
          if (input_length > 0) {
            log_beta.narrow(0, b, 1).narrow(1, input_length-1, 1).fill_(neginf);
            log_beta_a[input_length-1][2*target_length] = log_probs_a[input_length-1][BLANK];
            grad_a[input_length-1][BLANK] = log_alpha_a[input_length-1][2*target_length] + log_beta_a[input_length-1][2*target_length];

            if (target_length > 0) {
              auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 2*target_length-1, BLANK);
              log_beta_a[input_length-1][2*target_length-1] = log_probs_a[input_length-1][current_target_prime];

              // the first two are a blank and a non-blank, so we know they are different and we don't need to do log+
              grad_a[input_length-1][current_target_prime] = log_alpha_a[input_length-1][2*target_length-1] + log_beta_a[input_length-1][2*target_length-1];
            }
          }

          // now loop applying eq (10) / (11)
          for (i64 t=input_length-2; t>=0; t--) {
            // this loop over s could be parallel/vectorized and doesn't really need to be descending...
            // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
            // for the cuda implementation, that gave a speed boost.
            for (i64 s=2*target_length; s>=0; s--) {
              Scalar lb1 = log_beta_a[t+1][s];
              Scalar lbmax = lb1;
              Scalar lb2, lb3;
              auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
              if (s < 2*target_length) {
                lb2 = log_beta_a[t+1][s+1];
                if (lb2 > lbmax)
                  lbmax = lb2;
              } else {
                lb2 = neginf;
              }
              if ((s < 2*target_length-1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s+2, BLANK) !=
                                              current_target_prime)) {
                lb3 = log_beta_a[t+1][s+2];
                if (lb3 > lbmax)
                  lbmax = lb3;
              } else {
                lb3 = neginf;
              }
              if (lbmax == neginf)
                lbmax = 0;

              log_beta_a[t][s] = log(exp(lb1-lbmax)+exp(lb2-lbmax)+exp(lb3-lbmax))+lbmax + log_probs_a[t][current_target_prime];
              // one might check whether one can vectorize this better when done after the t-loop...
              // now that we have beta, we fill in the sum of alpha*beta in eq (16)
              // in contrast to the cuda implementation, we only parallelize over the batch, so we don't have a concurrency
              // issue (several s can map to the same target character)
              // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
              Scalar log_alpha_beta =  log_alpha_a[t][s] + log_beta_a[t][s];
              Scalar &lcab = grad_a[t][current_target_prime];
              if (lcab == neginf) {
                lcab = log_alpha_beta;
              } else {
                Scalar max = max(lcab, log_alpha_beta);
                lcab = log(exp(lcab-max)+exp(log_alpha_beta-max))+max;
              }
            }
          }

          // now grad has the sum of eq (16)
          // now we wrap up the calculation by adding in the remaining items of eq (16)
          // this could be a great target for further vectorization.
          // grad is the output gradient, nll is the loss. Note that the likelihood -nll is the Z of eq (16)
          Scalar gr =  grad_out.accessor<Scalar, 1>()[b];
          for (i64 t = 0; t < input_length; t++) { // or go for the full thing?
            for (i64 c = 0; c < num_labels; c++) {
              Scalar& res = grad_a[t][c];
              Scalar lp = log_probs_a[t][c];
              res = (exp(lp)-exp(res + nll - lp)) * gr;
            }
          }
          // zero the remainder
          if (input_length < max_input_length) {
            grad.narrow(0, input_length, max_input_length - input_length).narrow(1, b, 1).zero_();
          }
        }
      });
      return grad;
        */
}

pub fn ctc_loss_cpu(
        log_probs:      &Tensor,
        targets:        &Tensor,
        input_lengths:  &[i32],
        target_lengths: &[i32],
        BLANK:          i64,
        zero_infinity:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            (void)zero_infinity; // only used for backwards
      return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_cpu", [&] {
          if (targets.scalar_type() == kLong) {
            return ctc_loss_cpu_template<Scalar, kLong>(log_probs, targets, input_lengths, target_lengths, BLANK);
          } else {
            return ctc_loss_cpu_template<Scalar, kInt>(log_probs, targets, input_lengths, target_lengths, BLANK);
          }
      });
        */
}

pub fn ctc_loss_backward_cpu(
        grad:               &Tensor,
        log_probs:          &Tensor,
        targets:            &Tensor,
        input_lengths:      &[i32],
        target_lengths:     &[i32],
        neg_log_likelihood: &Tensor,
        log_alpha:          &Tensor,
        BLANK:              i64,
        zero_infinity:      bool) -> Tensor {
    
    todo!();
        /*
            return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_backward_cpu", [&] {
          if (targets.scalar_type() == kLong) {
            return ctc_loss_backward_cpu_template<Scalar,kLong>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
          } else {
            return ctc_loss_backward_cpu_template<Scalar,kInt>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
          }
      });
        */
}

/**
  | this wrapper function dispatches to the native
  | and cudnn implementations and hides the
  | alpha/grad from the user (by just returning the
  | loss)
  |
  | the gradient is implemented for _cudnn_ctc_loss
  | (just in derivatives.yaml) and _ctc_loss and
  | this function has automatic gradients
  |
  | it also handles the reduction if desired
  */
pub fn ctc_loss(
    log_probs:      &Tensor,
    targets:        &Tensor,
    input_lengths:  &[i32],
    target_lengths: &[i32],
    BLANK:          i64,
    reduction:      i64,
    zero_infinity:  bool) -> Tensor {

    todo!();
        /*
            bool use_cudnn =
          (log_probs.device().type() == kCUDA) &&
          _use_cudnn_ctc_loss(
              log_probs, targets, input_lengths, target_lengths, BLANK);

      Tensor res;
      if (use_cudnn) {
        // non-deterministic ctc loss on cudnn disabled due to inconsistent results
        // see: https://github.com/pytorch/pytorch/issues/21680
        res = get<0>(_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, BLANK, /*deterministic=*/true, zero_infinity));
      } else {
        // if the targets are on CPU (which you need for CuDNN, let's move them to
        // GPU as a service for the user)
        res = get<0>(_ctc_loss(
            log_probs,
            targets.to(log_probs.device(), kLong),
            input_lengths,
            target_lengths,
            BLANK,
            zero_infinity));
        if (zero_infinity) {
          res = where(res == Scalar(numeric_limits<double>::infinity()), zeros({}, res.options()), res);
        }
      }
      if (reduction == Reduction::Mean) {
        auto target_lengths_t =
            tensor(target_lengths, res.options()).clamp_min(1);
        return (res / target_lengths_t).mean();
      } else if (reduction == Reduction::Sum) {
        return res.sum();
      }
      return res;
        */
}

/**
  | Convenience function accepting Tensors
  |
  */
pub fn ctc_loss_tensor(
        log_probs:      &Tensor,
        targets:        &Tensor,
        input_lengths:  &Tensor,
        target_lengths: &Tensor,
        BLANK:          i64,
        reduction:      i64,
        zero_infinity:  bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false), "input_lengths must be integral");
      TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

      Tensor ilc = input_lengths.to(Device(kCPU), kLong).contiguous();
      Tensor tlc = target_lengths.to(Device(kCPU), kLong).contiguous();
      IntArrayRef il(ilc.data_ptr<i64>(), ilc.numel());
      IntArrayRef tl(tlc.data_ptr<i64>(), tlc.numel());
      return native::ctc_loss(log_probs, targets, il, tl, BLANK, reduction, zero_infinity);
        */
}
