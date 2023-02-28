crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/MultinomialKernel.cpp]

pub fn multinomial_with_replacement_apply<Scalar>(
        result:    &mut Tensor,
        self_:     &Tensor,
        n_sample:  i64,
        generator: Option<Generator>)  {

    todo!();
        /*
            auto gen = get_generator_or_default<CPUGeneratorImpl>(generator, getDefaultCPUGenerator());
      // See Note [Acquire lock when using random generators]
      lock_guard<mutex> lock(gen->mutex_);

      i64 n_categories = self.size(-1);
      i64 n_dist = self.dim() > 1 ? self.size(-2) : 1;

      /* cumulative probability distribution vector */
      Tensor cum_dist = empty({n_categories}, self.options());

      const Scalar * const self_ptr = self.data_ptr<Scalar>();
      Scalar * const cum_dist_ptr = cum_dist.data_ptr<Scalar>();
      i64 * const result_ptr = result.data_ptr<i64>();

      auto self_stride_0 = self.dim() > 1 ? self.stride(-2) : 0;
      auto self_stride_1 = self.stride(-1);

      auto cum_dist_stride_0 = cum_dist.stride(0);

      auto result_dist_stride_0 = result.dim() > 1 ? result.stride(-2) : 0;
      auto result_dist_stride_1 = result.stride(-1);

      for (i64 i = 0; i < n_dist; i++) {
        /* Get normalized cumulative distribution from prob distribution */
        Scalar sum = 0;
        Scalar val;
        int n_zeros = 0;
        for (i64 j = 0; j < n_categories; j++) {
          val = self_ptr[i * self_stride_0 + j * self_stride_1];
          TORCH_CHECK(val >= 0, "invalid multinomial distribution (encountering probability entry < 0)");
    // NB: isfinite doesn't bode well with libc++ for half datatypes,
    // so we manually cast it to a double and perform the check.
    #if defined(_LIBCPP_VERSION)
          TORCH_CHECK(isfinite(static_cast<double>(val)),
                      "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
    #else
          TORCH_CHECK(isfinite(val),
                      "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
    #endif

          sum += val;
          if (val == 0) {
            n_zeros += 1;
          }
          cum_dist_ptr[j * cum_dist_stride_0] = sum;
        }

        TORCH_CHECK(sum > 0, "invalid multinomial distribution (sum of probabilities <= 0)");

        /* normalize cumulative probability distribution so that last val is 1
        i.e. doesn't assume original self row sums to one */
        if ((sum > 0) || ((sum < 1.00001) && (sum > 0.99999))) {
          for (i64 j = 0; j < n_categories; j++) {
            cum_dist_ptr[j * cum_dist_stride_0] /= sum;
          }
        }

        for (i64 j = 0; j < n_sample; j++) {
          /* sample a probability mass from a uniform distribution */
          uniform_real_distribution<double> uniform(0, 1);
          double uniform_sample = uniform(gen);
          /* Do a binary search for the slot in which the prob falls
          ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
          int left_pointer = 0;
          int right_pointer = n_categories;
          int mid_pointer;
          Scalar cum_prob;
          int sample_idx;
          /* Make sure the last cumulative distribution bucket sums to 1 */
          cum_dist_ptr[(n_categories - 1) * cum_dist_stride_0] = 1;

          while(right_pointer - left_pointer > 0) {
            mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
            cum_prob = cum_dist_ptr[mid_pointer * cum_dist_stride_0];
            if (cum_prob < uniform_sample) {
              left_pointer = mid_pointer + 1;
            }
            else {
              right_pointer = mid_pointer;
            }
          }
          sample_idx = left_pointer;

          /* store in result tensor (will be incremented for lua compat by wrapper) */
          result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] = sample_idx;
        }
      }
        */
}

pub fn multinomial_with_replacement_kernel_impl(
        result:   &mut Tensor,
        self_:    &Tensor,
        n_sample: i64,
        gen:      Option<Generator>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "multinomial", [&] {
        multinomial_with_replacement_apply<Scalar>(result, self, n_sample, gen);
      });
        */
}

register_dispatch!{multinomial_with_replacement_stub, &multinomial_with_replacement_kernel_impl}
