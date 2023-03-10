crate::ix!();


pub enum QuantizationKind {

    /**
      | A simple quantization scheme that determines
      | quantization parameter by just looking
      | at min/max.
      |
      */
    MIN_MAX_QUANTIZATION,

    /// Minimizes L2 norm of quantization error
    L2_MIN_QUANTIZATION,

    /**
      | fast search to remove histogram outliers
      | and approximate L2 min
      |
      */
    L2_MIN_QUANTIZATION_APPROX,

    /// Minimizes Kullback-Leibler divergence
    KL_MIN_QUANTIZATION,

    /**
      | Take 99 percentail (only works with
      | sparsity preserving quantization)
      |
      */
    P99_QUANTIZATION,
    L1_MIN_QUANTIZATION,
}

/**
  | Represents a quantization scheme that
  | provides quantization parameter based
  | on distribution of data to be quantized.
  |
  */
pub struct QuantizationFactory {

    /**
      | precision used for activations in main
      | operations like matmul
      |
      */
    activation_precision:                 i32,

    /// precision used for weights
    weight_precision:                     i32,

    /**
      | precision used for the requantization
      | multiplier
      |
      */
    requantization_multiplier_precision:  i32,

    /// precision used for element-wise addition
    eltwise_quantize_precision:           i32,

    /// preserve zeros in quantization
    preserve_activation_sparsity:         bool,

    /// preserve zeros in quantization
    preserve_weight_sparsity:             bool,

    /// restrict scaling to a power of two
    force_scale_power_of_two:             bool,

    /**
      | P99 percentage to select out from the
      | full histogram for weights
      |
      */
    weight_p99_threshold:                 f32,

    /**
      | P99 percentage to select out from the
      | full histogram for activations
      |
      */
    activation_p99_threshold:             f32,

    activation_kind:                      QuantizationKind,

    weight_kind:                          QuantizationKind,
}

impl Default for QuantizationFactory {
    fn default() -> Self {
        Self {
            activation_precision:                 8,
            weight_precision:                     8,
            requantization_multiplier_precision:  32,
            eltwise_quantize_precision:           16,
            preserve_activation_sparsity:         false,
            preserve_weight_sparsity:             false,
            force_scale_power_of_two:             false,
            activation_kind:                      QuantizationKind::MIN_MAX_QUANTIZATION,
            weight_kind:                          QuantizationKind::MIN_MAX_QUANTIZATION,
            weight_p99_threshold:                 0.99,
            activation_p99_threshold:             0.99,
        }
    }
}

define_int32!{
    caffe2_dnnlowp_activation_quantization_precision,
    8,
    "Precision used for activation tensors"
}

define_int32!{
    caffe2_dnnlowp_weight_quantization_precision,
    8,
    "Precision used for weight tensors"
}

define_int32!{
    caffe2_dnnlowp_requantization_multiplier_precision,
    32,
    "Precision of integer multipliers used for rescaling quantized numbers"
}

define_int32!{
    caffe2_dnnlowp_eltwise_quantization_precision,
    16,
    "Precision used for intermediate numbers during elementwise operations"
}

define_bool!{
    caffe2_dnnlowp_force_scale_power_of_two,
    false,
    "When true, force quantization scales to a power of two"
}

define_bool!{
    caffe2_dnnlowp_preserve_activation_sparsity,
    false,
    "When true, 0 is mapped to 0 after quantization: i.e., symmetric quantization"
}

define_bool!{
    caffe2_dnnlowp_preserve_weight_sparsity,
    false,
    "When true, 0 is mapped to 0 after quantization: i.e., symmetric quantization"
}

define_string!{
    caffe2_dnnlowp_activation_quantization_kind,
    "min_max",
    "Quantization method for activation tensors. Allowed values: 
        min_max, l2, l2_approx, kl, l1, p99"
}

define_string!{
    caffe2_dnnlowp_weight_quantization_kind,
    "min_max",
    "Quantization method for weight tensors. Allowed values: 
        min_max, l2, l2_approx, kl, l1, p99"
}

define_double!{
    caffe2_dnnlowp_weight_p99_threshold,
    0.99,
    "P99 threshold to select out from the full histogram for weights."
}

define_double!{
    caffe2_dnnlowp_activation_p99_threshold,
    0.99,
    "P99 threshold to select out from the full histogram for activations."
}

define_int32!{
    caffe2_dnnlowp_nbits_in_non_outlier,
    8,
    "When outlier-aware quantization is used, if a quantized number can be 
        represented by this number of bits, it is considered not an outlier so 
        handled with 16-bit accumulation"
}

define_int32!{
    caffe2_dnnlowp_copy_to_32bit_frequency,
    32,
    "When outlier-aware quantization is used, this option specifies how often 
        we spill 16-bit accumulated numbers to 32-bit during the first pass"
}

define_bool!{
    caffe2_dnnlowp_force_slow_path,
    false,
    "When true, use slow path in quantization"
}

/**
  | Parse a string to QuantizationKind
  |
  */
#[inline] pub fn string_to_kind(s: &String) -> QuantizationKind {
    
    todo!();
    /*
        string s_lower(s);
      transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);

      if (s_lower == "min_max" || s == "MIN_MAX_QUANTIZATION") {
        return QuantizationFactory::MIN_MAX_QUANTIZATION;
      } else if (s_lower == "l1" || s == "L1_MIN_QUANTIZATION") {
        return QuantizationFactory::L1_MIN_QUANTIZATION;
      } else if (s_lower == "l2" || s == "L2_MIN_QUANTIZATION") {
        return QuantizationFactory::L2_MIN_QUANTIZATION;
      } else if (s_lower == "l2_approx" || s == "L2_MIN_QUANTIZATION_APPROX") {
        if (FLAGS_caffe2_dnnlowp_preserve_weight_sparsity ||
            FLAGS_caffe2_dnnlowp_preserve_activation_sparsity) {
          return QuantizationFactory::L2_MIN_QUANTIZATION;
        } else {
          return QuantizationFactory::L2_MIN_QUANTIZATION_APPROX;
        }
      } else if (s_lower == "kl" || s == "KL_MIN_QUANTIZATION") {
        return QuantizationFactory::KL_MIN_QUANTIZATION;
      } else if (s_lower == "p99" || s == "P99_QUANTIZATION") {
        return QuantizationFactory::P99_QUANTIZATION;
      } else {
        assert(false);
        return QuantizationFactory::MIN_MAX_QUANTIZATION;
      }
    */
}

impl QuantizationFactory {
    
    /**
      | Get the default factory whose policy
      | is determined by gflags
      |
      */
    #[inline] pub fn get_default_instance(&mut self) -> *mut QuantizationFactory {
        
        todo!();
        /*
            static QuantizationFactory singleton(
          FLAGS_caffe2_dnnlowp_activation_quantization_precision,
          FLAGS_caffe2_dnnlowp_weight_quantization_precision,
          FLAGS_caffe2_dnnlowp_requantization_multiplier_precision,
          FLAGS_caffe2_dnnlowp_eltwise_quantization_precision,
          FLAGS_caffe2_dnnlowp_preserve_activation_sparsity,
          FLAGS_caffe2_dnnlowp_preserve_weight_sparsity,
          FLAGS_caffe2_dnnlowp_force_scale_power_of_two,
          StringToKind(FLAGS_caffe2_dnnlowp_activation_quantization_kind),
          StringToKind(FLAGS_caffe2_dnnlowp_weight_quantization_kind),
          FLAGS_caffe2_dnnlowp_weight_p99_threshold,
          FLAGS_caffe2_dnnlowp_activation_p99_threshold);

      static bool log_printed = false;
      if (!log_printed) {
        LOG(INFO) << "activation_precision "
                  << FLAGS_caffe2_dnnlowp_activation_quantization_precision;
        LOG(INFO) << "weight_precision "
                  << FLAGS_caffe2_dnnlowp_weight_quantization_precision;
        LOG(INFO) << "requantization_multiplier_precision "
                  << FLAGS_caffe2_dnnlowp_requantization_multiplier_precision;
        LOG(INFO) << "eltwise_quantize_precision "
                  << FLAGS_caffe2_dnnlowp_eltwise_quantization_precision;
        LOG(INFO) << "preserve_activation_sparsity "
                  << FLAGS_caffe2_dnnlowp_preserve_activation_sparsity;
        LOG(INFO) << "preserve_weight_sparsity "
                  << FLAGS_caffe2_dnnlowp_preserve_weight_sparsity;
        LOG(INFO) << "force_scale_power_of_two "
                  << FLAGS_caffe2_dnnlowp_force_scale_power_of_two;
        LOG(INFO) << "activation_quantization_kind "
                  << FLAGS_caffe2_dnnlowp_activation_quantization_kind;
        LOG(INFO) << "weight_quantization_kind "
                  << FLAGS_caffe2_dnnlowp_weight_quantization_kind;
        LOG(INFO) << "weight p99 threshold  "
                  << FLAGS_caffe2_dnnlowp_weight_p99_threshold;
        LOG(INFO) << "activation p99 threshold  "
                  << FLAGS_caffe2_dnnlowp_activation_p99_threshold;
        LOG(INFO) << "nbits_in_non_outlier "
                  << FLAGS_caffe2_dnnlowp_nbits_in_non_outlier;
        LOG(INFO) << "copy_to_32bit_frequency "
                  << FLAGS_caffe2_dnnlowp_copy_to_32bit_frequency;
        LOG(INFO) << "omp_get_max_threads() " << caffe2::dnnlowp_get_max_threads();

        log_printed = true;
      }

      return &singleton;
        */
    }

    pub fn new(
        activation_precision:                i32,
        weight_precision:                    i32,
        requantization_multiplier_precision: i32,
        eltwise_quantize_precision:          i32,
        preserve_activation_sparsity:        bool,
        preserve_weight_sparsity:            bool,
        force_scale_power_of_two:            bool,
        activation_kind:                     QuantizationKind,
        weight_kind:                         QuantizationKind,
        weight_p99_threshold:                f32,
        activation_p99_threshold:            f32) -> Self {
    
        todo!();
        /*
            : activation_precision_(activation_precision),
          weight_precision_(weight_precision),
          requantization_multiplier_precision_(requantization_multiplier_precision),
          eltwise_quantize_precision_(eltwise_quantize_precision),
          preserve_activation_sparsity_(preserve_activation_sparsity),
          preserve_weight_sparsity_(preserve_weight_sparsity),
          force_scale_power_of_two_(force_scale_power_of_two),
          activation_kind_(activation_kind),
          weight_kind_(weight_kind),
          weight_p99_threshold_(weight_p99_threshold),
          activation_p99_threshold_(activation_p99_threshold)
        */
    }
    
    /**
      | Choose quantization based on histogram
      | of values to optimize the quantization
      | errors ignoring a few outliers
      |
      */
    #[inline] pub fn choose_quantization_params_with_histogram_and_precision(&self, 
        hist:              &Histogram,
        kind:              QuantizationKind,
        precision:         i32,
        preserve_sparsity: bool,
        is_weight:         Option<bool>) -> TensorQuantizationParams 
    {
        let is_weight: bool = is_weight.unwrap_or(false);

        todo!();
        /*
            switch (kind) {
        case L2_MIN_QUANTIZATION:
          return L2ErrorMinimization().ChooseQuantizationParams(
              hist, preserve_sparsity, precision);
        case L2_MIN_QUANTIZATION_APPROX:
          return L2ErrorMinimization().NonlinearQuantizationParamsSearch(
              hist, preserve_sparsity, precision);
        case L1_MIN_QUANTIZATION:
          return L1ErrorMinimization().ChooseQuantizationParams(
              hist, preserve_sparsity, precision);
        case KL_MIN_QUANTIZATION:
          return KLDivergenceMinimization().ChooseQuantizationParams(
              hist, preserve_sparsity, precision);
        case P99_QUANTIZATION:
          return P99(is_weight ? weight_p99_threshold_ : activation_p99_threshold_)
              .ChooseQuantizationParams(hist, preserve_sparsity, precision);
        case MIN_MAX_QUANTIZATION:
        default:
          return ChooseQuantizationParams(
              hist.Min(), hist.Max(), precision, preserve_sparsity);
      }
        */
    }
    
    #[inline] pub fn choose_quantization_params_with_histogram(
        &self, 
        hist: &Histogram, 
        is_weight: Option<bool>) -> TensorQuantizationParams 
    {
        let is_weight: bool = is_weight.unwrap_or(false);
        
        todo!();
        /*
            if (is_weight) {
        return ChooseQuantizationParams(
            hist,
            GetWeightKind(),
            GetWeightPrecision(),
            GetPreserveWeightSparsity(),
            true);
      } else {
        return ChooseQuantizationParams(
            hist,
            GetActivationKind(),
            GetActivationPrecision(),
            GetPreserveActivationSparsity(),
            false);
      }
        */
    }
    
    /**
      | Choose quantization based on the values
      | in an array to optimize the quantization
      | errors ignoring a few outliers
      |
      */
    #[inline] pub fn choose_quantization_params_with_values_and_precision(&self, 
        values:            *const f32,
        len:               i32,
        kind:              QuantizationKind,
        precision:         i32,
        preserve_sparsity: bool) -> TensorQuantizationParams {
        
        todo!();
        /*
            float min = 0, max = 0;
      fbgemm::FindMinMax(values, &min, &max, len);

      if (MIN_MAX_QUANTIZATION == kind) {
        return ChooseQuantizationParams(min, max, precision, preserve_sparsity);
      } else {
        if (0 == len) {
          return ChooseQuantizationParams(min, max, precision, preserve_sparsity);
        }

        /** Adjust the granularity of histogram collection to
         * the quantization precision. Use 8x more number of bins
         * in the histogram should be sufficient for linear quantization.
         */
        Histogram hist(1 << (precision + 3), min, max);
        for (int i = 0; i < len; ++i) {
          hist.Add(values[i]);
        }

        return ChooseQuantizationParams(hist, kind, precision, preserve_sparsity);
      }
        */
    }
    
    #[inline] pub fn choose_quantization_params_with_values_len_weight(&self, 
        values:    *const f32,
        len:       i32,
        is_weight: bool) -> TensorQuantizationParams {
        
        todo!();
        /*
            if (is_weight) {
        return ChooseQuantizationParams(
            values,
            len,
            GetWeightKind(),
            GetWeightPrecision(),
            GetPreserveWeightSparsity());
      } else {
        return ChooseQuantizationParams(
            values,
            len,
            GetActivationKind(),
            GetActivationPrecision(),
            GetPreserveActivationSparsity());
      }
        */
    }
    
    /**
      | Given a real_multiplier, produces
      | a pair (quantized_multiplier, right_shift)
      | where quantized_multiplier is an int32
      | representing a fixed-point value (in
      | practice we only produce positive values)
      | and right_shift is an amount to shift
      | right by, so that the floating-point
      | multiplication of some int32 input
      | value by real_multiplier,
      | 
      | return static_cast<int32>(int32_value
      | * real_multiplier);
      | 
      | is best approximated by the integer-arithmetic-only
      | code
      | 
      | return RoundingRightShift(
      | 
      | Multiplication(int32_value, quantized_multiplier),
      | right_shift);
      | 
      | -----------
      | @note
      | 
      | all this code only needs to run offline
      | to generate the quantized neural network
      | workload, not at runtime on the device
      | on which quantized neural networks
      | need to run. So it's not performance-critical
      | at all.
      |
      */
    #[inline] pub fn choose_requantization_multiplier(
        &self, 
        real_multiplier: f32, 
        target_qparams: TensorQuantizationParams) -> RequantizationParams 
    {
        todo!();
        /*
            RequantizationParams params;
      params.target_qparams = target_qparams;
      params.real_multiplier = real_multiplier;

      fbgemm::ChooseRequantizationMultiplier(
          real_multiplier,
          &params.multiplier,
          &params.right_shift,
          requantization_multiplier_precision_);

      return params;
        */
    }

    #[inline] pub fn choose_quantization_params_with_values(&self, 
        values:    *const f32,
        len:       i32,
        is_weight: Option<bool>) -> TensorQuantizationParams {
        let is_weight: bool = is_weight.unwrap_or(false);

        todo!();
        /*
        
        */
    }

    /**
      | Choose quantization scale and zero_point
      | that maps floating-point range [min, max]
      | to the integer range of the specified
      | precision
      */
    #[inline] pub fn choose_quantization_params_with_precision(&self, 
        min:               f32,
        max:               f32,
        precision:         i32,
        preserve_sparsity: bool,
        is_signed:         Option<bool>) -> TensorQuantizationParams {
        let is_signed: bool = is_signed.unwrap_or(false);

        todo!();
        /*
            TensorQuantizationParams qparams = fbgemm::ChooseQuantizationParams(
            min,
            max,
            is_signed ? -(1 << (precision - 1)) : 0,
            is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
            preserve_sparsity,
            force_scale_power_of_two_);
        qparams.precision = precision;
        return qparams;
        */
    }

    /**
      | Choose quantization scale and zero_point
      | that maps floating-point range [min, max]
      | to the default integer range of this
      | quantization factory
      */
    #[inline] pub fn choose_quantization_params(&self, 
        min:       f32,
        max:       f32,
        is_weight: Option<bool>) -> TensorQuantizationParams {
        let is_weight: bool = is_weight.unwrap_or(false);

        todo!();
        /*
            return ChooseQuantizationParams(
            min,
            max,
            is_weight ? GetWeightPrecision() : GetActivationPrecision(),
            is_weight ? GetPreserveWeightSparsity()
                      : GetPreserveActivationSparsity());
        */
    }

    #[inline] pub fn get_activation_precision(&self) -> i32 {
        
        todo!();
        /*
            return activation_precision_;
        */
    }
    
    #[inline] pub fn get_weight_precision(&self) -> i32 {
        
        todo!();
        /*
            return weight_precision_;
        */
    }
    
    #[inline] pub fn get_eltwise_quantize_precision(&self) -> i32 {
        
        todo!();
        /*
            return eltwise_quantize_precision_;
        */
    }
    
    #[inline] pub fn get_preserve_activation_sparsity(&self) -> bool {
        
        todo!();
        /*
            return preserve_activation_sparsity_;
        */
    }
    
    #[inline] pub fn get_preserve_weight_sparsity(&self) -> bool {
        
        todo!();
        /*
            return preserve_weight_sparsity_;
        */
    }
    
    #[inline] pub fn get_activation_kind(&self) -> QuantizationKind {
        
        todo!();
        /*
            return activation_kind_;
        */
    }
    
    #[inline] pub fn get_weight_kind(&self) -> QuantizationKind {
        
        todo!();
        /*
            return weight_kind_;
        */
    }
    
    #[inline] pub fn set_weight_p99threshold(&mut self, threshold: f32)  {
        
        todo!();
        /*
            weight_p99_threshold_ = threshold;
        */
    }
    
    #[inline] pub fn set_activation_p99threshold(&mut self, threshold: f32)  {
        
        todo!();
        /*
            activation_p99_threshold_ = threshold;
        */
    }
}

#[inline] pub fn adjust_hist_to_include_zero(
    hist: &Histogram,
    min:  *mut f32,
    max:  *mut f32) -> Vec<f32> {
    
    todo!();
    /*
        const vector<uint64_t> bins = *hist.GetHistogram();
      *min = hist.Min();
      *max = hist.Max();
      int nbins = bins.size();
      float bin_width = (*max - *min) / nbins;

      // Pad histogram to include zero
      int additional_nbins = 0;
      int offset = 0;
      if (*min > 0) {
        // additional nbins to include 0
        additional_nbins = ceil(*min / bin_width);
        offset = additional_nbins;
        *min -= additional_nbins * bin_width;
        assert(*min <= 0);
      } else if (*max < 0) {
        additional_nbins = ceil((-*max) / bin_width);
        *max += additional_nbins * bin_width;
        assert(*max >= 0);
      }

      vector<float> bins_f(nbins + additional_nbins);
      for (int i = 0; i < nbins; ++i) {
        bins_f[i + offset] = bins[i];
      }
      return bins_f;
    */
}
