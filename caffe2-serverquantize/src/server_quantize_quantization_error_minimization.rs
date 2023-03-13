crate::ix!();


pub trait QuantizationErrorMinimization {

    fn choose_quantization_params_impl(&mut self, 
        hist:              &Histogram,
        preserve_sparsity: bool,
        precision:         i32) -> TensorQuantizationParams;

    fn choose_quantization_params(&mut self, 
        hist:              &Histogram,
        preserve_sparsity: Option<bool>,
        precision:         Option<i32>) -> TensorQuantizationParams {

        let preserve_sparsity: bool = preserve_sparsity.unwrap_or(false);
        let precision:          i32 = precision.unwrap_or(8);

        self.choose_quantization_params_impl(hist, preserve_sparsity, precision)
    }
}

pub enum NormMinimizationKind {
    L1,
    L2,
}

///---------------
pub struct NormMinimization {
    kind:  NormMinimizationKind,
}

impl NormMinimization {

    pub fn new(kind: NormMinimizationKind) -> Self {
    
        todo!();
        /*
            : kind_(kind)
        */
    }
    
    /**
      Filter out outliers in input distributions
      Exploit the input distributions for the quick search

      Faster approximate search
      */
    #[inline] pub fn nonlinear_quantization_params_search(&mut self, 
        hist:              &Histogram,
        preserve_sparsity: Option<bool>,
        precision:         Option<i32>) -> TensorQuantizationParams {

        let preserve_sparsity: bool = preserve_sparsity.unwrap_or(false);
        let precision:          i32 = precision.unwrap_or(8);
        
        todo!();
        /*
            if (preserve_sparsity) {
        VLOG(2) << "l2_approx with symmetric quantization falls back to L2";
        return ChooseQuantizationParams(hist, preserve_sparsity, precision);
      }
      VLOG(2) << "Using the nonlinear quantile search";

      float min, max;
      vector<float> bins_f(dnnlowp::adjust_hist_to_include_zero(hist, &min, &max));
      int nbins = bins_f.size();
      float bin_width = (max - min) / nbins;
      float scale = (max - min) / float((1 << precision) - 1);
      if (bin_width == 0 || scale < SMALL_SCALE_THRESHOLD) {
        QuantizationFactory* qfactory = QuantizationFactory::GetDefaultInstance();
        return qfactory->ChooseQuantizationParams(
            min, max, precision, preserve_sparsity);
      }
      int dst_nbins = 1 << precision;

      float org_max = max;
      float org_min = min;
      // calculate the CDF
      uint64_t total = 0;
      for (uint64_t x : bins_f) {
        total += x;
      }
      vector<uint64_t> CDF;
      uint64_t sum = 0;
      for (uint64_t x : bins_f) {
        sum += x;
        CDF.push_back(sum);
      }

      double stepsize = 0.00001; // experiment on the granularity
      double alpha = 0.0f, beta = 1.0f; // lowerbound and upperbound
      int start_bin = 0;
      int end_bin = nbins - 1;
      double norm_min = double::max;

      while (alpha < beta) {
        // find the next step
        double next_alpha = alpha + stepsize;
        double next_beta = beta - stepsize;

        // find the left and right bins between the quantile bounds
        int i = start_bin, j = end_bin;
        while (i < end_bin && CDF[i] < next_alpha * total)
          i++;
        while (j > start_bin && CDF[j] > next_beta * total)
          j--;

        // decide the next move
        // cout << i << ", " << j << endl;
        int next_start_bin = start_bin, next_end_bin = end_bin;
        if ((i - start_bin) > (end_bin - j)) {
          // move the start_bin
          next_start_bin = i;
          alpha = next_alpha;
        } else {
          // move the end_bin
          next_end_bin = j;
          beta = next_beta;
        }

        if (next_start_bin == start_bin && next_end_bin == end_bin)
          continue;
        // calculate the norm
        double norm = 0;
        double dst_bin_width =
            bin_width * (next_end_bin - next_start_bin + 1) / dst_nbins;

        // go over each histogram bin and accumulate errors
        for (int src_bin = 0; src_bin < nbins; ++src_bin) {
          // distances from the beginning of first dst_bin to the beginning and
          // end of src_bin
          double src_bin_begin = (src_bin - next_start_bin) * bin_width;
          double src_bin_end = src_bin_begin + bin_width;

          // which dst_bins the beginning and end of src_bin belong to?
          int dst_bin_of_begin = std::min(
              (1 << precision) - 1.,
              std::max(0., floor(src_bin_begin / dst_bin_width)));
          int dst_bin_of_end = std::min(
              (1 << precision) - 1.,
              std::max(0., floor(src_bin_end / dst_bin_width)));

          double dst_bin_of_begin_center =
              dst_bin_of_begin * dst_bin_width + dst_bin_width / 2;
          double density = bins_f[src_bin] / bin_width;
          if (dst_bin_of_begin == dst_bin_of_end) {
            // if src_bin is entirely within 1 dst_bin
            double delta_begin = src_bin_begin - dst_bin_of_begin_center;
            double delta_end = src_bin_end - dst_bin_of_begin_center;
            norm += GetNorm(delta_begin, delta_end, density, kind_);
          } else {
            double delta_begin = src_bin_begin - dst_bin_of_begin_center;
            double delta_end = dst_bin_width / 2;
            norm += GetNorm(delta_begin, delta_end, density, kind_);

            norm += (dst_bin_of_end - dst_bin_of_begin - 1) *
                GetNorm(-dst_bin_width / 2, dst_bin_width / 2, density, kind_);

            double dst_bin_of_end_center =
                dst_bin_of_end * dst_bin_width + dst_bin_width / 2;
            delta_begin = -dst_bin_width / 2;
            delta_end = src_bin_end - dst_bin_of_end_center;
            norm += GetNorm(delta_begin, delta_end, density, kind_);
          }
        }
        if (norm > norm_min)
          break;
        norm_min = norm;
        start_bin = next_start_bin;
        end_bin = next_end_bin;
      }
      VLOG(2) << "best quantization range " << start_bin << "," << end_bin + 1
              << "," << norm_min;

      double selected_sum = 0;
      for (int i = start_bin; i < end_bin + 1; ++i) {
        selected_sum += bins_f[i];
      }
      VLOG(2) << "best quantization range covers "
              << (double)selected_sum / total * 100 << " %%";

      max = min + bin_width * (end_bin + 1);
      min = min + bin_width * start_bin;

      VLOG(2) << "Org min " << org_min << " org max " << org_max << " found min "
              << min << " max " << max << " with minimal norm " << norm_min;
      QuantizationFactory* qfactory = QuantizationFactory::GetDefaultInstance();
      return qfactory->ChooseQuantizationParams(
          min, max, precision, preserve_sparsity);
        */
    }
}

impl QuantizationErrorMinimization for NormMinimization {

    fn choose_quantization_params_impl(
        &mut self,
        hist:              &Histogram,
        preserve_sparsity: bool,
        precision:         i32) -> TensorQuantizationParams {
        self.nonlinear_quantization_params_search(hist,Some(preserve_sparsity),Some(precision))
    }
}


///-------------------------
pub struct L1ErrorMinimization {
    base: NormMinimization,
}

impl Default for L1ErrorMinimization {
    
    fn default() -> Self {
        todo!();
        /*
            : NormMinimization(L1
        */
    }
}

///-------------------------
pub struct P99 {
    threshold:  f32,
}

impl P99 {
    
    pub fn new(p99_threshold: Option<f32>) -> Self {
    
        let p99_threshold: f32 = p99_threshold.unwrap_or(0.99);

        todo!();
        /*
            : threshold_(p99_threshold)
        */
    }
}

impl QuantizationErrorMinimization for P99 {

    #[inline] fn choose_quantization_params_impl(&mut self, 
        hist:              &Histogram,
        preserve_sparsity: bool,
        precision:         i32) -> TensorQuantizationParams 
    {
        todo!();
        /*
          float min, max;
          std::vector<float> bins_f(
              dnnlowp::adjust_hist_to_include_zero(hist, &min, &max));
          int nbins = bins_f.size();
          CAFFE_ENFORCE(min <= 0.f);
          CAFFE_ENFORCE(max >= 0.f);
          float org_max = max;
          float org_min = min;
          float bin_width = (max - min) / nbins;
          int zero_bin = round(-min / bin_width);

          int best_width = 0;
          double total_sum = 0;
          for (int i = 0; i < nbins; ++i) {
            total_sum += bins_f[i];
          }
          double sum = 0;
          std::vector<double> CDF(nbins, 0.f);
          for (int i = 0; i < nbins; ++i) {
            sum += bins_f[i];
            CDF[i] = (double)sum / total_sum;
          }
          CAFFE_ENFORCE(threshold_ > 0.5 && threshold_ < 1);
          double left_quantile = (1.0f - threshold_) / 2.0f;
          double right_quantile = 1.0f - left_quantile;
          int i_begin = 0;
          int i_end = nbins - 2;
          bool finished = false;
          while (i_begin <= i_end && !finished) {
            finished = true;
            if (CDF[i_begin] < left_quantile) {
              i_begin++;
              finished = false;
            }
            if (CDF[i_end] > right_quantile) {
              finished = false;
              i_end--;
            }
          }
          min = i_begin * bin_width + org_min;
          max = (i_end + 2) * bin_width + org_min;

          VLOG(2) << "Org min " << org_min << " org max " << org_max << " found min "
                  << min << " max " << max;

          QuantizationFactory* qfactory = QuantizationFactory::GetDefaultInstance();
          return qfactory->ChooseQuantizationParams( min, max, precision, preserve_sparsity);
        */
    }
 
    #[inline] fn choose_quantization_params(&mut self, 
        hist:              &Histogram,
        preserve_sparsity: Option<bool>,
        precision:         Option<i32>) -> TensorQuantizationParams 
    {
        let preserve_sparsity: bool = preserve_sparsity.unwrap_or(true);
        let precision:          i32 = precision.unwrap_or(8);

        self.choose_quantization_params_impl(hist, preserve_sparsity, precision)
    }
}
