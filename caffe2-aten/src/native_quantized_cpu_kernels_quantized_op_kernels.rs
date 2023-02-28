crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp]

pub fn check_tensor_memory_format(
        ref_:  &Tensor,
        other: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          ref.is_contiguous(ref.suggest_memory_format()),
          "Quantized tensor should be contiguous");
      TORCH_CHECK(
          other.is_contiguous(ref.suggest_memory_format()),
          "Float tensor should be contiguous "
          "in same memory format as quantizd tensor");
        */
}

// ****************** HEY YOU! YES YOU! Read this! ********************
//
// Please read the README.md in this directory before editing this file
pub fn qcat_nhwc_kernel<const ReLUFused: bool = false>(
        qxs:        &List<Tensor>,
        dim:        i64,
        scale:      f64,
        zero_point: i64) -> Tensor {

    todo!();
        /*
            const Tensor& qx0 = qxs[0];
      i64 C_out = 0;
      vector<i64> Cs_in;
      // Prefix sum of input channels for fast indexing
      vector<i64> Cs_sum;
      vector<double> scales;
      vector<i64> zero_pts;
      vector<void*> data_ptrs;

      for (const Tensor& qx : qxs) {
        TORCH_CHECK(
            qx.dim() == qx0.dim(),
            "Tensors must have the same number of dimensions: got ",
            qx.dim(),
            " and ",
            qx0.dim());
    #define CHECK_DIM(d)                                            \
      TORCH_CHECK(                                                  \
          qx.size(d) == qx0.size(d),                                \
          "Sizes of tensors must match expect in dimension 1. Got", \
          qx.size(d),                                               \
          " and ",                                                  \
          qx0.size(d));
        CHECK_DIM(0);
        CHECK_DIM(2);
        CHECK_DIM(3);
        TORCH_CHECK(
            qx.scalar_type() == qx0.scalar_type(),
            "Expected object of scalar type ",
            toString(qx0.scalar_type()),
            " but got scalar type ",
            toString(qx.scalar_type()));
        Cs_in.push_back(qx.size(1));
        Cs_sum.push_back(C_out);
        C_out += qx.size(1);
        scales.push_back(qx.q_scale());
        zero_pts.push_back(qx.q_zero_point());
        data_ptrs.push_back(qx.data_ptr());
      }

      const i64 N = qx0.size(0);
      const i64 H = qx0.size(2);
      const i64 W = qx0.size(3);
      float inv_scale = 1.0 / scale;

      auto output = _empty_affine_quantized(
          {N, C_out, H, W},
          qx0.options().memory_format(MemoryFormat::ChannelsLast),
          scale,
          zero_point,
          nullopt);

      // N, H, and W are explicitly captured here because there's a bug in GCC5
      // which causes an internal compiler error if they're not
      AT_DISPATCH_QINT_TYPES(output.scalar_type(), "qcat_nhwc", [&, N, H, W]() {
        using Vec = Vectorized<Scalar>;
        for (i64 batch = 0; batch < N; ++batch) {
          for (i64 row = 0; row < H; ++row) {
            for (i64 col = 0; col < W; ++col) {
              // loop over input tensors
              for (const auto tidx : irange(Cs_in.size())) {
                Scalar::underlying* optr =
                    reinterpret_cast<Scalar::underlying*>(output.data_ptr()) +
                    batch * H * W * C_out + row * W * C_out + col * C_out +
                    Cs_sum[tidx];

                auto curr_C = Cs_in[tidx];
                float curr_scale = scales[tidx];
                i64 curr_zero_pt = zero_pts[tidx];

                Scalar::underlying* iptr =
                    reinterpret_cast<Scalar::underlying*>(data_ptrs[tidx]) +
                    batch * H * W * curr_C + row * W * curr_C + col * curr_C;

                constexpr i64 VLEN = Vec::size();
                i64 c = 0;

                // Vectorized loop
                if (c + VLEN <= curr_C) {
                  auto curr_scale_vec = Vectorized<float>(curr_scale);
                  auto curr_zero_pt_vec = Vectorized<float>((float)curr_zero_pt);
                  auto scale_neg_zp_premul = curr_scale_vec * curr_zero_pt_vec.neg();
                  for (; c + VLEN <= curr_C; c += VLEN) {
                    auto inp_vec = Vec::loadu(iptr + c);
                    auto float_values = inp_vec.dequantize(
                        curr_scale_vec, curr_zero_pt_vec, scale_neg_zp_premul);
                    Vec::float_vec_return_type retvals;
                    for (int i = 0; i < Vec::float_num_vecs(); ++i) {
                      if (ReLUFused) {
                        retvals[i] =
                            vec::maximum(float_values[i], Vectorized<float>(0.0f));
                      } else {
                        retvals[i] = float_values[i];
                      }
                    }
                    auto quantized =
                        Vec::quantize(retvals, scale, zero_point, inv_scale);
                    quantized.store(optr + c);
                  }
                }

                // Scalar loop
                for (; c < curr_C; ++c) {
                  auto float_val = native::dequantize_val(
                      curr_scale,
                      curr_zero_pt,
                      reinterpret_cast<Scalar*>(iptr)[c]);
                  if (ReLUFused) {
                    float_val = max(0.0f, float_val);
                  }
                  optr[c] = native::quantize_val<Scalar>(
                                scale, zero_point, float_val)
                                .val_;
                } // for c

              } // for tidx
            } // for col
          } // for row
        } // for b
      });

      return output;
        */
}

/**
  | horizontal sum over a range of values
  |
  */
pub fn hsum<H: CanHSum>(
    A:   *const H,
    len: usize) -> i64 {

    /**
      | horizontal sum over a range of u8
      |
      */
    fn hsum_u8(
            A:   *const u8,
            len: i32) -> i64 {
        
        todo!();
            /*
                i64 row_sum = 0;
          int i = 0;

        #ifdef target_feature = "avx2"
          __m256i sum_v = _mm256_setzero_si256();
          __m256i one_epi16_v = _mm256_set1_epi16(1);
          __m256i one_epi8_v = _mm256_set1_epi8(1);
          // vectorized
          for (; i < len / 32 * 32; i += 32) {
            __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
            sum_v = _mm256_add_epi32(
              sum_v,
              _mm256_madd_epi16(
                // first argument is unsigned, second is signed
                _mm256_maddubs_epi16(src_v, one_epi8_v),
              one_epi16_v)
            );
          }

          alignas(64) i32 temp[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
          for (int k = 0; k < 8; ++k) {
            row_sum += temp[k];
          }
        #endif // target_feature = "avx2"

          // scalar
          for (; i < len; ++i) {
            row_sum += A[i];
          }

          return row_sum;
            */
    }

    /// horizontal sum over a range of i8
    ///
    fn hsum_i8(
            A:   *const i8,
            len: i32) -> i64 {
        
        todo!();
            /*
                i64 row_sum = 0;
          int i = 0;

        #ifdef target_feature = "avx2"
          __m256i sum_v = _mm256_setzero_si256();
          __m256i one_epi16_v = _mm256_set1_epi16(1);
          __m256i one_epi8_v = _mm256_set1_epi8(1);
          // vectorized
          for (; i < len / 32 * 32; i += 32) {
            __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
            sum_v = _mm256_add_epi32(
              sum_v,
              _mm256_madd_epi16(
                // first argument is unsigned, second is signed
                _mm256_maddubs_epi16(one_epi8_v, src_v),
              one_epi16_v)
            );
          }

          alignas(64) i32 temp[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
          for (int k = 0; k < 8; ++k) {
            row_sum += temp[k];
          }
        #endif // target_feature = "avx2"

          // scalar
          for (; i < len; ++i) {
            row_sum += A[i];
          }

          return row_sum;
            */
    }

    /// horizontal sum over a range of i32
    ///
    fn hsum_i32(
        A:   *const i32,
        len: i32) -> i64 {
        
        todo!();
            /*
                i64 row_sum = 0;
          int i = 0;

        #ifdef target_feature = "avx2"
          __m256i sum_epi64 = _mm256_setzero_si256();
          // vectorized
          for (; i < len / 8 * 8; i += 8) {
            __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
            // widen
            __m128i src_lo_epi32 = _mm256_castsi256_si128(src_epi32);
            __m128i src_hi_epi32 = _mm256_extractf128_si256(src_epi32, 1);
            __m256i src_lo_epi64 = _mm256_cvtepi32_epi64(src_lo_epi32);
            __m256i src_hi_epi64 = _mm256_cvtepi32_epi64(src_hi_epi32);
            // add
            sum_epi64 = _mm256_add_epi64(sum_epi64, src_lo_epi64);
            sum_epi64 = _mm256_add_epi64(sum_epi64, src_hi_epi64);
          }

          alignas(64) i64 temp[4];
          _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_epi64);
          for (int k = 0; k < 4; ++k) {
            row_sum += temp[k];
          }
        #endif // target_feature = "avx2"

          // scalar
          for (; i < len; ++i) {
            row_sum += A[i];
          }

          return row_sum;
            */
    }

    todo!("need to write the type dispatch for HSum");
}

pub trait HSum {}
pub trait HSumSq {}

impl HSum for i8 {}
impl HSum for u8 {}
impl HSum for i32 {}

impl HSumSq for i8 {}
impl HSumSq for u8 {}
impl HSumSq for i32 {}

pub fn hsum_sq<H: HSumSq>(
    A:   *const H,
    len: i32) -> i64 {

    /**
      | horizontal sum of squares over a range
      | of u8
      |
      */
    fn hsum_sq_u8(
            A:   *const u8,
            len: i32) -> i64 {
        
        todo!();
            /*
                i64 row_sum = 0;
          int i = 0;

        #ifdef target_feature = "avx2"
          // vectorized
          __m256i sum_v_epu32 = _mm256_setzero_si256();
          alignas(64) i32 temp[8];
          int overflow_threshold = 262144; // 2147483647(max of int32)/(256*256)*8 = 262144
          int loop = len / overflow_threshold + 1;
          for(int j=0; j<=loop; j++){
            for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
              // (i15, ..., i0)
              __m128i src_epu8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
              __m256i src_epu16 = _mm256_cvtepu8_epi16(src_epu8);
              // (i15 ^ 2, ..., i0 ^ 2)
              __m256i sq_epu16 = _mm256_mullo_epi16(src_epu16, src_epu16);
              // (i7 ^ 2, ..., i0 ^ 2)
              __m128i sq_lo_epu16 = _mm256_castsi256_si128(sq_epu16);
              // (i15 ^ 2, ..., i8 ^ 2)
              __m128i sq_hi_epu16 = _mm256_extractf128_si256(sq_epu16, 1);
              // widen to epu32
              __m256i sq_lo_epu32 = _mm256_cvtepu16_epi32(sq_lo_epu16);
              __m256i sq_hi_epu32 = _mm256_cvtepu16_epi32(sq_hi_epu16);
              // add to running sum
              sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_lo_epu32);
              sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_hi_epu32);
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epu32);
            for (int k = 0; k < 8; ++k) {
              row_sum += temp[k];
            }
            sum_v_epu32 = _mm256_setzero_si256();
          }
        #endif // target_feature = "avx2"

          // scalar
          for (; i < len; ++i) {
            row_sum += A[i] * A[i];
          }

          return row_sum;
            */
    }

    /**
      | horizontal sum of squares over a range
      | of i8
      |
      */
    fn hsum_sq_i8(
            A:   *const i8,
            len: i32) -> i64 {
        
        todo!();
            /*
                i64 row_sum = 0;
          int i = 0;

        #ifdef target_feature = "avx2"
          // vectorized
          __m256i sum_v_epi32 = _mm256_setzero_si256();
          alignas(64) i32 temp[8];

          int overflow_threshold = 1048576; //2147483647/(128*128)*8 = 1048576
          int loop = len / overflow_threshold + 1;

          for(int j=0; j<=loop; j++){
            for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
              // (i15, ..., i0)
              __m128i src_epi8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
              __m256i src_epi16 = _mm256_cvtepi8_epi16(src_epi8);
              // (i15 ^ 2, ..., i0 ^ 2)
              __m256i sq_epi16 = _mm256_mullo_epi16(src_epi16, src_epi16);
              // (i7 ^ 2, ..., i0 ^ 2)
              __m128i sq_lo_epi16 = _mm256_castsi256_si128(sq_epi16);
              // (i15 ^ 2, ..., i8 ^ 2)
              __m128i sq_hi_epi16 = _mm256_extractf128_si256(sq_epi16, 1);
              // widen to epi32
              __m256i sq_lo_epi32 = _mm256_cvtepi16_epi32(sq_lo_epi16);
              __m256i sq_hi_epi32 = _mm256_cvtepi16_epi32(sq_hi_epi16);
              // add to running sum
              sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_lo_epi32);
              sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_hi_epi32);
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epi32);

            for (int k = 0; k < 8; ++k) {
              row_sum += temp[k];
            }
            sum_v_epi32 = _mm256_setzero_si256();
          }
        #endif // target_feature = "avx2"

          // scalar
          for (; i < len; ++i) {
            row_sum += A[i] * A[i];
          }

          return row_sum;
            */
    }

    /**
      | horizontal sum os squares over a range of
      | i32
      |
      | floats throughout are necessary to prevent
      | overflow
      */
    fn hsum_sq_i32(
        A:   *const i32,
        len: i32) -> f32 {
        
        todo!();
            /*
                float row_sum = 0;
          int i = 0;

        #ifdef target_feature = "avx2"
          __m256 sum_ps = _mm256_setzero_ps();
          // vectorized
          for (; i < len / 8 * 8; i += 8) {
            __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
            __m256 src_ps = _mm256_cvtepi32_ps(src_epi32);
            sum_ps = _mm256_add_ps(sum_ps, _mm256_mul_ps(src_ps, src_ps));
          }

          alignas(64) float temp[8];
          _mm256_store_ps(temp, sum_ps);
          for (int k = 0; k < 8; ++k) {
            row_sum += static_cast<float>(temp[k]);
          }
        #endif // target_feature = "avx2"

          // scalar
          for (; i < len; ++i) {
            i64 cur = static_cast<i64>(A[i]);
            row_sum += (float)cur * (float)cur;
          }

          return row_sum;
            */
    }

    todo!("write dispatch for HSumSq");
}

pub fn qrelu_kernel(
        qx: &Tensor,
        qy: &mut Tensor)  {
    
    todo!();
        /*
            const auto zero_point = qx.q_zero_point();
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
            qx.q_scale(),
            qx.q_zero_point(),
            nullopt);
        using Vec = Vectorized<Scalar>;
        auto zero_point_vec = Vec(Scalar(zero_point));
        auto iter = TensorIterator::unary_op(qy, qx);
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              return Scalar(max<underlying_t>(value.val_, zero_point));
            },
            [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
      });
        */
}

pub fn qrelu6_kernel(
        qx: &Tensor,
        qy: &mut Tensor)  {
    
    todo!();
        /*
            const auto zero_point = qx.q_zero_point();
      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu6", [&]() {
        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
            qx.q_scale(),
            qx.q_zero_point(),
            nullopt);
        using Vec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(qy, qx);
        Scalar six = native::quantize_val<Scalar>(
            qx.q_scale(), qx.q_zero_point(), 6.0);
        auto zero_point_vec = Vec(Scalar(zero_point));
        auto six_vec = Vec(six);
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              underlying_t relu_val =
                  max<underlying_t>(value.val_, zero_point);
              return Scalar(min<underlying_t>(relu_val, six.val_));
            },
            [&](Vec val) -> Vec { return val.relu6(zero_point_vec, six_vec); });
      });
        */
}

pub fn leaky_qrelu_out_kernel(
        out:    &mut Tensor,
        qx:     &Tensor,
        negval: &Scalar)  {
    
    todo!();
        /*
            i64 i_zp = qx.q_zero_point();
      float i_scale = qx.q_scale();

      i64 o_zp = out.q_zero_point();
      float o_scale = out.q_scale();
      float o_inv_scale = 1.0f / o_scale;

      float negval = negval_.to<float>();

      AT_DISPATCH_QINT_TYPES(out.scalar_type(), "leaky_qrelu", [&] {
        using Vec = Vectorized<float>;  // Naive implementation uses dequant/quant loop.
        using qVec = Vectorized<Scalar>;
        Vec zero_vec = Vec(0.0f);
        Vec one_vec = Vec(1.0f);

        Vec i_scale_vec = Vec((float)i_scale);
        Vec i_zp_vec = Vec((float)i_zp);
        Vec i_scale_zp_neg_premul_vec = i_scale_vec * i_zp_vec.neg();

        Vec negval_vec = Vec(negval);

        auto iter = TensorIterator::unary_op(out, qx);

        cpu_kernel_vec(
            iter,
            [&](Scalar value_qx) -> Scalar {
              auto value_dx = native::dequantize_val(i_scale, i_zp, value_qx);
              auto value_dy = value_dx > 0 ? value_dx : value_dx * negval;
              return native::quantize_val<Scalar>(o_scale, o_zp, value_dy);
            },
            [&](qVec qx_vec) -> qVec {
              /* Vectorized implementation creates a multiplicand vector, which has
               * "alpha" for all negative dx values and ones-vector for all
               * positive values of dx. The multiplicand then is multiplied by the
               * input.
               */
              auto dx_vec_vec = qx_vec.dequantize(i_scale_vec, i_zp_vec,
                                                  i_scale_zp_neg_premul_vec);
              for (auto& dx_vec: dx_vec_vec) {
                const auto multiplicand = Vec::blendv(negval_vec, one_vec,
                                                      dx_vec > zero_vec);
                dx_vec = dx_vec * multiplicand;
              }
              return qVec::quantize(dx_vec_vec, o_scale, o_zp, o_inv_scale);
            });
      });
        */
}

pub fn qsigmoid_kernel(
        qx:                &Tensor,
        qy:                &mut Tensor,
        output_scale:      f64,
        output_zero_point: i64)  {
    
    todo!();
        /*
            i64 zero_point = qx.q_zero_point();
      float scale = qx.q_scale();
      auto scale_vec = Vectorized<float>(scale);
      auto zero_point_vec = Vectorized<float>((float)zero_point);
      auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
        float inv_output_scale = 1.0 / output_scale;

        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
            output_scale,
            output_zero_point,
            nullopt);
        auto iter = TensorIterator::unary_op(qy, qx);

        using Vec = Vectorized<Scalar>;
        cpu_kernel_vec(
            iter,
            [&](Scalar value_qx) -> Scalar {
              const auto value_dx =
                  native::dequantize_val(scale, zero_point, value_qx);
              const auto value_dy = 1.0f / (1.0 + exp((-value_dx)));
              return native::quantize_val<Scalar>(
                  output_scale, output_zero_point, value_dy);
            },
            [&](Vec value_qx) -> Vec {
              auto value_dx = value_qx.dequantize(
                  scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
              for (auto& value: value_dx) {
                value = value.neg();
                value = value.exp();
                value = Vectorized<float>(1.0f) + value;
                value = value.reciprocal();
              }
              return Vec::quantize(
                  value_dx, output_scale, output_zero_point, inv_output_scale);
            });
      });
        */
}

pub fn qhardsigmoid_kernel(
        qx: &Tensor,
        qy: &mut Tensor)  {
    
    todo!();
        /*
            i64 zero_point = qx.q_zero_point();
      float scale = qx.q_scale();
      auto scale_vec = Vectorized<float>(scale);
      auto zero_point_vec = Vectorized<float>((float)zero_point);
      auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardsigmoid", [&]() {

        // - Output scale is set to 1.0 / 2^(BIT_NUM)
        float output_scale = 0.00390625;  // 1.0 / 2^8
        if (SCALAR_TYPE == kQInt32) {
          output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
        }
        float inv_output_scale = 1.0 / output_scale;

        // The default zero-point is zero.  As a one-off optimization for
        // kQInt8, we set the zero-point to -128 to maximize precision in the
        // [0, 1] output range. kQInt32 can be handled in a future PR if needed.
        i64 output_zero_point = 0;
        if (SCALAR_TYPE == kQInt8) {
          output_zero_point = -128;
        }

        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU).dtype(SCALAR_TYPE),
            output_scale,
            output_zero_point,
            qx.suggest_memory_format());
        auto iter = TensorIterator::unary_op(qy, qx);

        using qVec = Vectorized<Scalar>;
        using fVec = Vectorized<float>;
        fVec kZeroVec(0.0f);
        fVec kThreeVec(3.0f);
        fVec kSixVec(6.0f);

        // Naive implemenentation: uses dequantize/execute/quantize routine
        cpu_kernel_vec(
            iter,
            [&](Scalar qx) -> Scalar {
              auto x = native::dequantize_val(scale, zero_point, qx);
              const auto y = min(max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
              return native::quantize_val<Scalar>(
                  output_scale, output_zero_point, y);
            },
            [&](qVec value_qx) -> qVec {
              auto value_dx = value_qx.dequantize(
                  scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
              for (auto& value : value_dx) {
                value =
                    vec::minimum(
                        vec::maximum(value + kThreeVec, kZeroVec),
                        kSixVec) /
                    kSixVec;
              }
              return qVec::quantize(
                  value_dx, output_scale, output_zero_point, inv_output_scale);
            });
      });
        */
}


pub fn qclamp_kernel(
        qx:         &Tensor,
        min_scalar: &Scalar,
        max_scalar: &Scalar,
        qy:         &mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
            qx.q_scale(),
            qx.q_zero_point(),
            nullopt);
        using Vec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(qy, qx);
        auto min = min_scalar.to<float>();
        auto max = max_scalar.to<float>();
        Scalar min_q = native::quantize_val<Scalar>(
            qx.q_scale(), qx.q_zero_point(), min);
        Scalar max_q = native::quantize_val<Scalar>(
            qx.q_scale(), qx.q_zero_point(), max);
        auto min_vec = Vec(min_q);
        auto max_vec = Vec(max_q);
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              underlying_t min_clamped =
                  max<underlying_t>(value.val_, min_q.val_);
              return Scalar(min<underlying_t>(min_clamped, max_q.val_));
            },
            [&](Vec val) -> Vec {
              auto min_clamped = val.maximum(min_vec);
              return min_clamped.minimum(max_vec);
            });
      });
        */
}


pub fn qclamp_min_kernel(
        qx:         &Tensor,
        min_scalar: &Scalar,
        qy:         &mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU)
                .dtype(SCALAR_TYPE)
                .memory_format(qx.suggest_memory_format()),
            qx.q_scale(),
            qx.q_zero_point(),
            nullopt);
        using Vec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(qy, qx);
        auto min = min_scalar.to<float>();
        Scalar min_q = native::quantize_val<Scalar>(
            qx.q_scale(), qx.q_zero_point(), min);
        auto min_vec = Vec(min_q);
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              return Scalar(max<underlying_t>(value.val_, min_q.val_));
            },
            [&](Vec val) -> Vec { return val.maximum(min_vec); });
      });
        */
}


pub fn qclamp_max_kernel(
        qx:         &Tensor,
        max_scalar: &Scalar,
        QY:         &mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU)
                .dtype(SCALAR_TYPE)
                .memory_format(qx.suggest_memory_format()),
            qx.q_scale(),
            qx.q_zero_point(),
            nullopt);
        using Vec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(qy, qx);
        auto max = max_scalar.to<float>();
        Scalar max_q = native::quantize_val<Scalar>(
            qx.q_scale(), qx.q_zero_point(), max);
        auto max_vec = Vec(max_q);
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              return Scalar(min<underlying_t>(value.val_, max_q.val_));
            },
            [&](Vec val) -> Vec { return val.minimum(max_vec); });
      });
        */
}

pub fn qthreshold_kernel(
    // TODO: For future tasks, since output quantization parameters are set equal to
    // the input ones, it might make sense to implement this completely in the
    // quantized domain.
    qx:               &Tensor,
    threshold_scalar: &Scalar,
    value_scalar:     &Scalar,
    qy:               &mut Tensor)  {
    
    todo!();
        /*
            // defines input and output scales and zero_points
      i64 input_zero_point = qx.q_zero_point();
      float input_scale = qx.q_scale();
      i64 output_zero_point = qy.q_zero_point();
      float output_scale = qy.q_scale();
      float inv_output_scale = 1.0 / output_scale;

      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qthreshold", [&]() {
        qy = _empty_affine_quantized(
          qx.sizes(),
          device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
          qx.q_scale(),
          qx.q_zero_point(),
          nullopt);

        // vectorized
        using Vec = Vectorized<float>;
        using qVec = Vectorized<Scalar>;
        // defines the iterator
        auto iter = TensorIterator::unary_op(qy, qx);
        // defines the vectorized versions
        Vec input_scale_vec = Vec(input_scale);
        Vec input_zero_point_vec = Vec(input_zero_point);
        Vec input_scale_neg_zp_premul_vec = input_scale_vec * input_zero_point_vec.neg();
        // defines the floating-point versions of threshold and value
        float threshold_float = threshold_scalar.to<float>();
        float value_float = value_scalar.to<float>();
        Vec threshold_vec = Vec(threshold_float);
        Vec value_vec = Vec(value_float);

        // Naive implemenentation: uses dequantize/execute/quantize routine
        cpu_kernel_vec(
            iter,
            [&](Scalar value_qx) -> Scalar {
              // dequantize
              const auto x = native::dequantize_val(input_scale, input_zero_point, value_qx);
              // Applies the Threshold operation
              const auto y = x > threshold_float ? x : value_float;
              // quantize
              return native::quantize_val<Scalar>(output_scale, output_zero_point, y);
            },
            [&](qVec value_qx) -> qVec {
              // dequantize
              auto dx_vec = value_qx.dequantize(
                input_scale_vec, input_zero_point_vec, input_scale_neg_zp_premul_vec);
              for (auto& value : dx_vec) {
                // check if any elements are below threshold
                const auto cmp_to_threshold = value > threshold_vec;
                if (cmp_to_threshold.zero_mask()) {
                  // blend
                  value = Vec::blendv(value_vec, value, cmp_to_threshold);
                }
              }
              // quantize
              return qVec::quantize(dx_vec, output_scale, output_zero_point, inv_output_scale);
            });
      });
        */
}


pub fn qhardswish_kernel(
        qx: &Tensor,
        qy: &mut Tensor)  {
    
    todo!();
        /*
            const auto i_scale = qx.q_scale();
      const auto i_zero_point = qx.q_zero_point();

      const auto o_scale = qy.q_scale();
      const auto o_zero_point = qy.q_zero_point();
      const float o_inv_scale = 1.0 / o_scale;

      using fVec = Vectorized<float>;
      fVec i_scale_vec(i_scale);
      fVec i_zero_point_vec(i_zero_point);
      fVec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();
      fVec zero_vec(0.0f);
      fVec three_vec(3.0f);
      fVec six_vec(6.0f);

      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardswish", [&]() {
        using qVec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(qy, qx);
        cpu_kernel_vec(
            iter,
            [&](Scalar value) -> Scalar {
              const auto x =
                  native::dequantize_val(i_scale, i_zero_point, value);
              const auto y = x * min(max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
              return native::quantize_val<Scalar>(o_scale, o_zero_point, y);
            },
            [&](qVec value) -> qVec {
              auto value_dx = value.dequantize(i_scale_vec, i_zero_point_vec,
                                               i_scale_neg_zp_premul_vec);
              for (auto& value: value_dx) {
                value = value * vec::minimum(
                  vec::maximum(value + three_vec, zero_vec),
                  six_vec
                ) / six_vec;
              }
              return qVec::quantize(value_dx, o_scale, o_zero_point, o_inv_scale);
            });
      });
        */
}


pub fn qtanh_kernel(
        qx: &Tensor,
        qy: &mut Tensor)  {
    
    todo!();
        /*
            i64 zero_point = qx.q_zero_point();
      float scale = qx.q_scale();
      auto scale_vec = Vectorized<float>(scale);
      auto zero_point_vec = Vectorized<float>((float)zero_point);
      auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qtanh", [&]() {
        // Naive implemenentation: uses dequantize/execute/quantize routine
        // - Output scale is set to 2.0 / 2^(BIT_NUM)
        // - For signed types output zero point is set to 0
        // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
        float output_scale = 0.0078125;  // 2.0 / 512
        i64 output_zero_point = 0;
        if (SCALAR_TYPE == kQInt32) {
          output_scale = 4.656612873077393e-10;  // 2.0 / 2^32
        } else if (SCALAR_TYPE == kQUInt8) {
          output_zero_point = 128;
        }
        float inv_output_scale = 1.0 / output_scale;

        qy = _empty_affine_quantized(
            qx.sizes(),
            device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
            output_scale,
            output_zero_point,
            nullopt);
        auto iter = TensorIterator::unary_op(qy, qx);

        using Vec = Vectorized<Scalar>;
        cpu_kernel_vec(
            iter,
            [&](Scalar value_qx) -> Scalar {
              const auto value_dx =
                  native::dequantize_val(scale, zero_point, value_qx);
              return native::quantize_val<Scalar>(
                  output_scale, output_zero_point, tanh(value_dx));
            },
            [&](Vec value_qx) -> Vec {
              const auto value_dx = value_qx.dequantize(
                  scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
              Vec::float_vec_return_type retvals;
              for (const auto idx : irange(Vec::float_num_vecs())) {
                retvals[idx] = value_dx[idx].tanh();
              }
              return Vec::quantize(
                  retvals, output_scale, output_zero_point, inv_output_scale);
            });
      });
        */
}


pub fn qelu_kernel(
        qx:          &Tensor,
        alpha:       &Scalar,
        scale:       &Scalar,
        input_scale: &Scalar,
        qy:          &mut Tensor)  {
    
    todo!();
        /*
            // scale and input_scale arguments refer to a generalized ELU formula
      // if x >= 0, ELU(x) = x * scale
      // if x <= 0, ELU(x) = (exp(x * input_scale) - 1) * scale
      // in the normal ELU formula, both are equal to 1
      // they are NOT related to the quantization scale term

      i64 i_zp = qx.q_zero_point();
      float i_scale = qx.q_scale();

      // In a future PR, we can improve on output scale and zero_point
      // selection.
      i64 o_zp = qy.q_zero_point();
      float o_scale = qy.q_scale();
      float inv_o_scale = 1.0 / o_scale;

      float alpha_float = alpha.to<float>();
      float scale_coef = scale.to<float>();
      float input_scale_coef = input_scale.to<float>();

      AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qelu_kernel", [&] {

        auto iter = TensorIterator::unary_op(qy, qx);

        // vectorized
        using Vec = Vectorized<float>;
        using qVec = Vectorized<Scalar>;

        Vec zero_vec = Vec(0.0f);
        Vec one_vec = Vec(1.0f);
        Vec alpha_vec = Vec(alpha_float);
        Vec scale_coef_vec = Vec(scale_coef);
        Vec input_scale_coef_vec = Vec(input_scale_coef);
        Vec i_scale_vec = Vec(i_scale);
        Vec i_zero_point_vec = Vec((float)i_zp);
        Vec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();

        cpu_kernel_vec(
          iter,
          [&](Scalar value_qx) -> Scalar {
            // dequantize
            const auto x = native::dequantize_val(i_scale, i_zp, value_qx);
            // ELU
            const auto y = x >= 0
              ? x * scale_coef
              : ((exp(x * input_scale_coef) - 1) * alpha_float * scale_coef);

            // quantize
            return native::quantize_val<Scalar>(o_scale, o_zp, y);
          },
          [&](qVec value_qx) -> qVec {
            // dequantize
            auto dx_vec_vec = value_qx.dequantize(i_scale_vec, i_zero_point_vec,
                                                i_scale_neg_zp_premul_vec);
            for (auto& value : dx_vec_vec) {
              // quickly check if any elements are below zero
              const auto cmp_to_zero = value > zero_vec;

              if (cmp_to_zero.zero_mask()) {
                Vec dx_vec_copy_neg_elu = value * one_vec;
                // calculate the negative part of ELU on the copy
                dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * input_scale_coef_vec;
                dx_vec_copy_neg_elu = dx_vec_copy_neg_elu.exp();
                dx_vec_copy_neg_elu = dx_vec_copy_neg_elu - one_vec;
                dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * alpha_vec;
                // blend
                value = Vec::blendv(dx_vec_copy_neg_elu, value,
                                            value > zero_vec);
              }

              value = value * scale_coef_vec;
            }
            // quantize
            return qVec::quantize(dx_vec_vec, o_scale, o_zp, inv_o_scale);
          }
        );

      });
        */
}

/**
  | Note: out is assumed to be the same size as
  | self and other.
  |
  | Note: Addition is only supported when self and
  | out are of the same dtype.
  |
  | Note: other is already assumed to be in int32,
  | i.e., it's round(float/self_scale)
  */
pub fn qadd_scalar_kernel<const ReLUFused: bool = false>(
        out:   &mut Tensor,
        self_: &Tensor,
        other: &Scalar)  {

    todo!();
        /*
            i64 zero_point = out.q_zero_point();
      float scale = out.q_scale();
      float inv_scale = 1.0f / scale;
      i64 self_zero_point = self.q_zero_point();
      float self_scale = self.q_scale();

      float multiplier = self_scale * inv_scale;

      AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
        using Vec = Vectorized<Scalar>;
        auto iter = TensorIterator::unary_op(out, self);
        auto other_val = other.to<i32>();
        auto other_vec = Vectorized<qint32>(static_cast<qint32>(other_val));
        cpu_kernel_vec(
            iter,
            [&](Scalar a) -> Scalar {
              i32 a_sub_z = static_cast<i32>(a.val_) -
                  static_cast<i32>(self_zero_point);
              i32 c = a_sub_z + other_val;
              Scalar res = native::requantize_from_int<Scalar>(
                  multiplier, zero_point, c);
              if (ReLUFused) {
                res.val_ = max<Scalar::underlying>(res.val_, zero_point);
              }
              return res;
            },
            [&](Vec a) -> Vec {
              Vec::int_vec_return_type a_sub_z =
                  a.widening_subtract(Vec(static_cast<Scalar>(self_zero_point)));
              Vec::int_vec_return_type c;
              for (const auto i : irange(Vec::int_num_vecs())) {
                c[i] = a_sub_z[i] + other_vec;
              }
              Vec rv = Vec::requantize_from_int(c, multiplier, zero_point);
              if (ReLUFused) {
                rv = rv.maximum(Vec(static_cast<Scalar>(zero_point)));
              }
              return rv;
            });
      });
        */
}

/**
  | Note: out is assumed to be the same size as
  | self and other.
  |
  | Note: Addition is only supported when self,
  | other, out are of the same dtype.
  */
pub fn qadd_kernel<const ReLUFused: bool = false>(
        out:   &mut Tensor,
        self_: &Tensor,
        other: &Tensor)  {

    todo!();
        /*
            i64 zero_point = out.q_zero_point();
      float scale = out.q_scale();
      float inv_scale = 1.0f / scale;
      i64 self_zero_point = self.q_zero_point();
      float self_scale = self.q_scale();
      i64 other_zero_point = other.q_zero_point();
      float other_scale = other.q_scale();

      // Broadcast out the parameters here to amortize out that cost across
      // loop iterations.
      // TODO: we can optimize dequantization by doing a premultiplication
      // of the zero point by scale and doing FMA on scale*x_q - (scale*zero_point)
      auto self_zero_point_vec = Vectorized<float>((float)self_zero_point);
      auto self_scale_vec = Vectorized<float>(self_scale);
      auto other_zero_point_vec = Vectorized<float>((float)other_zero_point);
      auto other_scale_vec = Vectorized<float>(other_scale);

      auto self_scale_neg_zp_premul_vec = self_scale_vec * self_zero_point_vec.neg();
      auto other_scale_zp_premul_vec = other_scale_vec * other_zero_point_vec.neg();

      auto iter = TensorIterator::borrowing_binary_op(out, self, other);

      AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
        using Vec = Vectorized<Scalar>;
        cpu_kernel_vec(
            iter,
            [&](Scalar a, Scalar b) -> Scalar {
              const auto da =
                  native::dequantize_val(self_scale, self_zero_point, a);
              const auto db =
                  native::dequantize_val(other_scale, other_zero_point, b);
              float c = da + db;
              if (ReLUFused) {
                c = max<float>(c, 0.0);
              }
              return native::quantize_val<Scalar>(scale, zero_point, c);
            },
            [&](Vec a, Vec b) -> Vec {
              const auto da = a.dequantize(
                  self_scale_vec, self_zero_point_vec, self_scale_neg_zp_premul_vec);
              const auto db = b.dequantize(
                  other_scale_vec, other_zero_point_vec, other_scale_zp_premul_vec);
              Vec::float_vec_return_type retvals;
              for (const auto i : irange(Vec::float_num_vecs())) {
                auto c = da[i] + db[i];
                if (ReLUFused) {
                  c = vec::maximum(c, Vectorized<float>(0.0f));
                }
                retvals[i] = c;
              }
              // TODO: fbgemm::Quantize doesn't support taking in the
              // pre-broadcasted parameters. We might be able to save some cycles by
              // enabling that in the API.
              // TODO: specialize fbgemm::Quantize for a single vector and make it
              // inlineable. This could help with interleaving as suggested by the
              // TensorIterator implementations
              auto rv = Vec::quantize(retvals, scale, zero_point, inv_scale);
              return rv;
            });
      });
        */
}

/**
  | Note: out is assumed to be the same size as
  | self and other.
  |
  | Note: Multiplication is only supported when
  | self, other, out are of the same dtype.
  */
pub fn qmul_kernel<const ReLUFused: bool = false>(
        out:   &mut Tensor,
        self_: &Tensor,
        other: &Tensor)  {

    todo!();
        /*
            i64 zero_point = out.q_zero_point();
      float scale = out.q_scale();
      float inv_scale = 1.0f / scale;
      i64 self_zero_point = self.q_zero_point();
      float self_scale = self.q_scale();
      i64 other_zero_point = other.q_zero_point();
      float other_scale = other.q_scale();

      float multiplier = self_scale * other_scale * inv_scale;

      auto iter = TensorIterator::borrowing_binary_op(out, self, other);

      AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul", [&]() {
        using Vec = Vectorized<Scalar>;
        cpu_kernel_vec(
            iter,
            [&](Scalar a, Scalar b) -> Scalar {
              i32 a_sub_z = static_cast<i32>(a.val_) -
                  static_cast<i32>(self_zero_point);
              i32 b_sub_z = static_cast<i32>(b.val_) -
                  static_cast<i32>(other_zero_point);
              i32 c = a_sub_z * b_sub_z;
              Scalar res = native::requantize_from_int<Scalar>(
                  multiplier, zero_point, c);
              if (ReLUFused) {
                res.val_ = max<Scalar::underlying>(res.val_, zero_point);
              }
              return res;
            },
            [&](Vec a, Vec b) -> Vec {
              Vec::int_vec_return_type a_sub_zp =
                  a.widening_subtract(Vec(static_cast<Scalar>(self_zero_point)));
              Vec::int_vec_return_type b_sub_zp =
                  b.widening_subtract(Vec(static_cast<Scalar>(other_zero_point)));
              Vec::int_vec_return_type c;
              for (const auto i : irange(Vec::int_num_vecs())) {
                c[i] = a_sub_zp[i] * b_sub_zp[i];
              }
              Vec rv = Vec::requantize_from_int(c, multiplier, zero_point);
              if (ReLUFused) {
                rv = rv.maximum(Vec(static_cast<Scalar>(zero_point)));
              }
              return rv;
            });
      });
        */
}

pub fn qmaxpool_2d_nhwc_kernel(
        qx: &Tensor,

        // input/output channels
        ic: i64,
        ih: i64,

        // input sizes
        iw: i64,
        oh: i64,

        // output sizes
        ow: i64,
        kh: i64,

        // kernel size
        kw: i64,
        sh: i64,

        // strides
        sw: i64,
        ph: i64,

        // padding
        pw: i64,
        dh: i64,

        // dilation
        dw: i64,
        qy: &mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d_nhwc", [&]() {
        Scalar* idata = static_cast<Scalar*>(qx.data_ptr());
        Scalar* odata = static_cast<Scalar*>(qy.data_ptr());

        // Loop over N
        for (i64 b = 0; b < qx.size(0); ++b) {
          // Loop over H
          auto* i_p =
              reinterpret_cast<Scalar::underlying*>(idata + b * iW * iH * iC);
          for (i64 row = 0; row < oH; ++row) {
            // Loop over W
            for (i64 col = 0; col < oW; ++col) {
              // Pointer to output data for this specific N,H,W position
              auto* o_p = reinterpret_cast<Scalar::underlying*>(
                  odata + b * oH * oW * iC + row * oW * iC + col * iC);

              // Loop over reduction block
              i64 h_start = row * sH - pH;
              i64 w_start = col * sW - pW;
              i64 h_end = min(h_start + (kH - 1) * dH + 1, iH);
              i64 w_end = min(w_start + (kW - 1) * dW + 1, iW);
              while (h_start < 0)
                h_start += dH;
              while (w_start < 0)
                w_start += dW;

              i64 c = 0;

              // Interleaved vector loop 4x
              constexpr auto vec_width = Vectorized<Scalar>::size();
              for (; c + 4 * vec_width <= iC; c += 4 * vec_width) {
                Vectorized<Scalar> acc{
                    Scalar(numeric_limits<Scalar::underlying>::lowest())};
                Vectorized<Scalar> accs[4] = {acc, acc, acc, acc};
                i64 tcntr = 0;
                i64 x, y;
                for (y = h_start; y < h_end; y += dH) {
                  for (x = w_start; x < w_end; x += dW) {
                    for (int i = 0; i < 4; ++i) {
                      tcntr = y * iW + x;
                      auto vals = Vectorized<Scalar>::loadu(
                          i_p + tcntr * iC + c + Vectorized<Scalar>::size() * i);
                      accs[i] = vec::maximum(accs[i], vals);
                    }
                  } // for x
                } // for y
                for (int i = 0; i < 4; ++i) {
                  accs[i].store(o_p + c + Vectorized<Scalar>::size() * i);
                }
              } // for c

              // Vector loop
              for (; c + vec_width <= iC; c += vec_width) {
                Vectorized<Scalar> acc{
                    Scalar(numeric_limits<Scalar::underlying>::lowest())};
                i64 tcntr = 0;
                i64 x, y;
                for (y = h_start; y < h_end; y += dH) {
                  for (x = w_start; x < w_end; x += dW) {
                    tcntr = y * iW + x;
                    auto vals = Vectorized<Scalar>::loadu(i_p + tcntr * iC + c);
                    acc = vec::maximum(acc, vals);
                  } // for x
                } // for y
                acc.store(o_p + c);
              } // for c

              for (; c < iC; ++c) {
                auto max_val = numeric_limits<Scalar::underlying>::lowest();
                i64 tcntr = 0;
                i64 x, y;
                for (y = h_start; y < h_end; y += dH) {
                  for (x = w_start; x < w_end; x += dW) {
                    tcntr = y * iW + x;
                    auto val = *(i_p + tcntr * iC + c);
                    max_val = max(max_val, val);
                  } // for x
                } // for y

                o_p[c] = max_val;
              } // for c
            } // for col
          } // for row
        } // for b
      });
        */
}




pub fn do_avg_pool_nhwc_on_avx2<T>(
        i_p:                     *const T::underlying,
        o_p:                     *mut T::underlying,
        c_start:                 &mut i32,
        input_zero_point_m_size: i32,
        output_zero_point:       i32,
        multiplier:              f32,
        dstart:                  i32,
        dend:                    i32,
        hstart:                  i32,
        hend:                    i32,
        wstart:                  i32,
        wend:                    i32,
        dsize:                   i32,
        hsize:                   i32,
        wsize:                   i32,
        csize:                   i32)  {

    todo!();
        /*
            #if defined(target_feature = "avx2") && !defined(_MSC_VER)
      // buffer for channel accumulator, used to interchange channel-loop
      // to inner-most, so that memory access of the input tensor data is
      // continuous.
      constexpr int cb_size = 16;
      constexpr int vec_width = Vectorized<T>::size() / 4;
      constexpr int cb_step = cb_size * vec_width;
      Vectorized<i32> acc_buffer[cb_size];
      Vectorized<float> acc_buffer_fp[cb_size];

      if (vec_width == 8) {
        for (int c = c_start; c < csize; c += cb_step) {
          int cend = min(cb_size, (csize - c) / vec_width);
          // initialize loop
          for (int ic = 0; ic < cend; ic++) {
            acc_buffer[ic] = Vectorized<i32>(input_zero_point_m_size);
          }
          // compute loop
          for (int id = dstart; id < dend; id++) {
            for (int ih = hstart; ih < hend; ih++) {
              for (int iw = wstart; iw < wend; iw++) {
                const int i_idx =
                    (id * wsize * hsize + ih * wsize + iw) *
                        csize +
                    c;
                for (int ic = 0; ic < cend; ic++) {
                  auto vals = vec::convert_to_int32<typename T::underlying>(
                      i_p + i_idx + ic * vec_width);
                  acc_buffer[ic] = acc_buffer[ic] + vals;
                }
              }
            }
          }
          // convert int32 accumulative to fp32
          vec::convert((int*)acc_buffer, (float*)acc_buffer_fp, cend * vec_width);

          // first quantize using AVX using 32 lanes, then 8, finally falls
          // back to single
          QuantizeAvx2<T>(
              (float*)acc_buffer_fp,
              o_p + c,
              cend * vec_width,
              multiplier,
              output_zero_point);
        }
        c_start = csize / vec_width * vec_width;
      }
    #endif
        */
}



pub fn do_avg_pool_on_avx2<T>(
        i_p:                     *mut T::underlying,
        o_p:                     *mut T::underlying,
        c:                       &mut i64,
        channel_size:            i64,
        channel_multiplier:      i64,
        input_zero_point_m_size: i32,
        output_zero_point:       i32,
        multiplier:              f32,
        dstart:                  i64,
        dend:                    i64,
        hstart:                  i64,
        hend:                    i64,
        wstart:                  i64,
        wend:                    i64,
        stride_c:                i64,
        stride_d:                i64,
        stride_h:                i64,
        stride_w:                i64)  {

    todo!();
        /*
            #if defined(target_feature = "avx2") && !defined(_MSC_VER)
      constexpr auto vec_width = Vectorized<T>::size() / 4;
      if (vec_width == 8) {
        for (; c + vec_width <= channel_size; c += vec_width) {
          i64 tcntr = 0;

          Vectorized<i32> acc(input_zero_point_m_size);
          for (i64 id = dstart; id < dend; id++) {
            for (i64 ih = hstart; ih < hend; ih++) {
              for (i64 iw = wstart; iw < wend; iw++) {
                tcntr = id * stride_D + ih * stride_H + iw * stride_W;
                auto vals = vec::convert_to_int32<typename T::underlying>(
                    i_p + tcntr * channel_multiplier + c * stride_C);
                acc = acc + vals;
              }
            }
          }
          i32 acc_int[vec_width];
          float acc_fp[vec_width];
          acc.store(acc_int);
          vec::convert(acc_int, acc_fp, vec_width);
          native::quantize_vec<T>(
              1.0f / multiplier,
              output_zero_point,
              acc_fp,
              reinterpret_cast<T*>(o_p + c),
              vec_width);
        }
      }
    #endif
        */
}


pub fn qadaptive_avg_pool_kernel(
        fn_name:  &String,
        qx:       &Tensor,
        qy:       &mut Tensor,
        b:        i64,
        sizec:    i64,

        // Set to 1 for 2d
        isized:   i64,

        isizeh:   i64,
        isizew:   i64,

        // Set to 1 for 2d
        osized:   i64,
        osizeh:   i64,
        osizew:   i64,
        istrideb: i64,
        istridec: i64,

        // Set to 1 for 2d
        istrided: i64,
        istrideh: i64,
        istridew: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(qx.scalar_type(), fn_name, [&]() {
        Scalar* idata = static_cast<Scalar*>(qx.data_ptr());
        Scalar* odata = static_cast<Scalar*>(qy.data_ptr());
        auto* i_p =
            reinterpret_cast<typename Scalar::underlying*>(idata + b * istrideB);

        float input_scale = qx.q_scale();
        float output_scale = qy.q_scale();
        int input_zero_point = qx.q_zero_point();
        int output_zero_point = qy.q_zero_point();

        for (i64 od = 0; od < osizeD; od++) {
          int istartD = (int)floor((float)(od * isizeD) / osizeD);
          int iendD = (int)ceil((float)((od + 1) * isizeD) / osizeD);
          int kD = iendD - istartD;
          for (i64 oh = 0; oh < osizeH; oh++) {
            int istartH = (int)floor((float)(oh * isizeH) / osizeH);
            int iendH = (int)ceil((float)((oh + 1) * isizeH) / osizeH);
            int kH = iendH - istartH;
            for (i64 ow = 0; ow < osizeW; ow++) {
              auto* o_p = reinterpret_cast<typename Scalar::underlying*>(
                  odata +
                  b * osizeD * osizeH * osizeW * sizeC +
                  od * osizeH * osizeW * sizeC +
                  oh * osizeW * sizeC +
                  ow * sizeC);
              int istartW = (int)floor((float)(ow * isizeW) / osizeW);
              int iendW = (int)ceil((float)((ow + 1) * isizeW) / osizeW);
              int kW = iendW - istartW;
              int size = kD * kH * kW;
              float multiplier = input_scale / output_scale / size;
              int input_zero_point_m_size = -input_zero_point * size;
              i64 c = 0;
              // For int8 or uint8quantization, we implicitly use int32 as
              // accumulation Or else, it will go to the slow path
              // TODO: support 16bit, 32bit, and etc.
              auto* internal_i_p = i_p +
                                   istartD * istrideD +
                                   istartH * istrideH +
                                   istartW * istrideW;

              // Note: If AVX is not available, `do_avg_pool_on_AVX2 is a noop.
              //       In that case, the following loop takes over
              // TODO: more vectorization with loop interleaving
              do_avg_pool_on_AVX2<Scalar>(
                  internal_i_p,
                  o_p,
                  c,
                  sizeC,
                  1,
                  input_zero_point_m_size,
                  output_zero_point,
                  multiplier,
                  0,
                  kD,
                  0,
                  kH,
                  0,
                  kW,
                  istrideC,
                  istrideD,
                  istrideH,
                  istrideW);

              // 1) The following loop handles the remaining channels
              // 2) It also handles the Non-AVX2 path
              for (; c < sizeC; ++c) {
                i32 acc_int32 = input_zero_point_m_size;
                i64 tcntr = 0;
                for (i64 id = 0; id < kD; ++id) {
                  for (i64 ih = 0; ih < kH; ++ih) {
                    for (i64 iw = 0; iw < kW; ++iw) {
                      tcntr = id * istrideD +
                              ih * istrideH +
                              iw * istrideW;
                      auto val = *(internal_i_p + tcntr + c * istrideC);
                      acc_int32 += val;
                    }
                  }
                }
                // clamp
                o_p[c] = native::quantize_val<Scalar>(1.0f / multiplier,
                                                            output_zero_point,
                                                            acc_int32).val_;
              } // c
            } // oh
          } // ow
        } // od
      });
        */
}



pub fn qadaptive_avg_pool2d_nhwc_kernel(
        qx:       &Tensor,
        qy:       &mut Tensor,
        b:        i64,
        sizec:    i64,
        isizeh:   i64,
        isizew:   i64,
        osizeh:   i64,
        osizew:   i64,
        istrideb: i64,
        istridec: i64,
        istrideh: i64,
        istridew: i64)  {
    
    todo!();
        /*
            _qadaptive_avg_pool_kernel("adaptive_avg_pool2d_nhwc",
                                 qx,
                                 qy,
                                 b,
                                 sizeC,
                                 /*isizeD=*/1,
                                 isizeH,
                                 isizeW,
                                 /*osizeD=*/1,
                                 osizeH,
                                 osizeW,
                                 istrideB,
                                 istrideC,
                                 /*istrideD=*/1,
                                 istrideH,
                                 istrideW);
        */
}


pub fn qadaptive_avg_pool3d_ndhwc_kernel(
        qx:       &Tensor,
        qy:       &mut Tensor,
        b:        i64,
        sizec:    i64,
        isized:   i64,
        isizeh:   i64,
        isizew:   i64,
        osized:   i64,
        osizeh:   i64,
        osizew:   i64,
        istrideb: i64,
        istridec: i64,
        istrided: i64,
        istrideh: i64,
        istridew: i64)  {
    
    todo!();
        /*
            _qadaptive_avg_pool_kernel("adaptive_avg_pool3d_ndhwc",
                                 qx,
                                 qy,
                                 b,
                                 sizeC,
                                 isizeD,
                                 isizeH,
                                 isizeW,
                                 osizeD,
                                 osizeH,
                                 osizeW,
                                 istrideB,
                                 istrideC,
                                 istrideD,
                                 istrideH,
                                 istrideW);
        */
}


pub fn qavg_pool_nhwc_kernel(
        fn_name:           &String,
        qx:                &Tensor,
        qy:                &mut Tensor,
        b:                 i64,
        n_input_plane:     i64,
        input_width:       i64,
        input_height:      i64,
        input_depth:       i64,
        output_width:      i64,
        output_height:     i64,
        output_depth:      i64,
        kw:                i32,
        kh:                i32,
        kd:                i32,
        dw:                i32,
        dh:                i32,
        dd:                i32,
        padw:              i32,
        padh:              i32,
        padd:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(qx.scalar_type(), fn_name, [&]() {
        Scalar* idata = static_cast<Scalar*>(qx.data_ptr());
        Scalar* odata = static_cast<Scalar*>(qy.data_ptr());
        int strideC = 1;
        int strideW = strideC * nInputPlane;
        int istrideH = strideW * inputWidth;
        int istrideD = istrideH * inputHeight;
        int istrideB = istrideD * inputDepth;
        int ostrideH = strideW * outputWidth;
        int ostrideD = ostrideH * outputHeight;
        int ostrideB = ostrideD * outputDepth;
        auto* i_p =
            reinterpret_cast<typename Scalar::underlying*>(idata + b * istrideB);

        // lift these operations outside the loop to reduce access overheads
        float input_scale = qx.q_scale();
        float output_scale = qy.q_scale();
        int input_zero_point = qx.q_zero_point();
        int output_zero_point = qy.q_zero_point();
        i64 divisor_override_factor =
            divisor_override.has_value() ? divisor_override.value() : 0;

        for (int od = 0; od < outputDepth; od++) {
          for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
              auto* o_p = reinterpret_cast<typename Scalar::underlying*>(
                  odata + b * ostrideB + od * ostrideD + oh * ostrideH +
                  ow * strideW);
              int dstart = od * dD - padD;
              int hstart = oh * dH - padH;
              int wstart = ow * dW - padW;

              int dend = min(dstart + kD, (int)inputDepth + padD);
              int hend = min(hstart + kH, (int)inputHeight + padH);
              int wend = min(wstart + kW, (int)inputWidth + padW);
              int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);

              dstart = max(dstart, 0);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              dend = min(dend, (int)inputDepth);
              hend = min(hend, (int)inputHeight);
              wend = min(wend, (int)inputWidth);

              int size = (dend - dstart) * (hend - hstart) * (wend - wstart);
              int divide_size = count_include_pad ? pool_size : size;
              int divide_factor =
                  divisor_override_factor ? divisor_override_factor : divide_size;
              float multiplier = input_scale / output_scale / divide_factor;
              int input_zero_point_m_size = -input_zero_point * size;

              int c_start = 0;

              // For int8 quantization, we implicitly use int32 as accumulation
              // Or else, it will go to the slow path
              // TODO: support 16bit, 32bit, and etc.
              do_avg_pool_nhwc_on_AVX2<Scalar>(
                  i_p,
                  o_p,
                  c_start,
                  input_zero_point_m_size,
                  output_zero_point,
                  multiplier,
                  dstart,
                  dend,
                  hstart,
                  hend,
                  wstart,
                  wend,
                  inputDepth,
                  inputHeight,
                  inputWidth,
                  nInputPlane);

              // 1) The following loop handles the remaining channels
              // 2) It also handles the Non-AVX2 path
              for (int c = c_start; c < nInputPlane; ++c) {
                i32 acc_int32 = input_zero_point_m_size;
                for (i64 id = dstart; id < dend; id++) {
                  for (i64 ih = hstart; ih < hend; ih++) {
                    for (i64 iw = wstart; iw < wend; iw++) {
                      auto val =
                          *(i_p + id * istrideD + ih * istrideH + iw * strideW +
                            c * strideC);
                      acc_int32 += val;
                    }
                  }
                }
                double acc_fp = acc_int32 * 1.0;
                // clamp
                o_p[c] = native::quantize_val<Scalar>(
                             1.0f / multiplier, output_zero_point, acc_fp)
                             .val_;
              } // c
            } // ow
          } // oh
        } // od
      });
        */
}


pub fn qavg_pool2d_nhwc_kernel(
        qx:                &Tensor,
        qy:                &mut Tensor,
        b:                 i64,
        n_input_plane:     i64,
        input_width:       i64,
        input_height:      i64,
        output_width:      i64,
        output_height:     i64,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {
    
    todo!();
        /*
            _qavg_pool_nhwc_kernel(
          "avg_pool2d_nhwc",
          qx,
          qy,
          b,
          nInputPlane,
          inputWidth,
          inputHeight,
          1,
          outputWidth,
          outputHeight,
          1,
          kW,
          kH,
          1,
          dW,
          dH,
          1,
          padW,
          padH,
          0,
          count_include_pad,
          divisor_override);
        */
}



pub fn qavg_pool3d_nhwc_kernel(
        qx:                &Tensor,
        qy:                &mut Tensor,
        b:                 i64,
        n_input_plane:     i64,
        input_width:       i64,
        input_height:      i64,
        input_depth:       i64,
        output_width:      i64,
        output_height:     i64,
        output_depth:      i64,
        kw:                i32,
        kh:                i32,
        kd:                i32,
        dw:                i32,
        dh:                i32,
        dd:                i32,
        padw:              i32,
        padh:              i32,
        padd:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {
    
    todo!();
        /*
            _qavg_pool_nhwc_kernel(
          "avg_pool3d_nhwc",
          qx,
          qy,
          b,
          nInputPlane,
          inputWidth,
          inputHeight,
          inputDepth,
          outputWidth,
          outputHeight,
          outputDepth,
          kW,
          kH,
          kD,
          dW,
          dH,
          dD,
          padW,
          padH,
          padD,
          count_include_pad,
          divisor_override);
        */
}




pub fn do_quantized_bilinear_on_avx2<T>(
        pos1:              &mut *const T::underlying,
        pos2:              &mut *mut T::underlying,
        input_height:      i64,
        input_width:       i64,
        output_height:     i64,
        output_width:      i64,
        channels:          i64,
        output_zero_point: i32,
        input_zero_point:  i32,
        inverse_scale:     f32,
        h0lambda:          f32,
        h1lambda:          f32,
        w0lambda:          f32,
        w1lambda:          f32,
        h1p:               i64,
        w1p:               i64) -> i64 {

    todo!();
        /*
            i64 c = 0;
    #if defined(target_feature = "avx2") && !defined(_MSC_VER)
      constexpr auto vec_width = Vectorized<T>::size() / 4;
      if (vec_width == 8) {
        for (; c + vec_width <= channels; c += vec_width) {
          Vectorized<float> pos1_fp_v[4];
          Vectorized<i32> pos1_int_v[4];
          pos1_int_v[0] = vec::convert_to_int32<typename T::underlying>(pos1);
          pos1_int_v[1] = vec::convert_to_int32<typename T::underlying>(
              pos1 + w1p * channels);
          pos1_int_v[2] = vec::convert_to_int32<typename T::underlying>(
              pos1 + h1p * input_width * channels);
          pos1_int_v[3] = vec::convert_to_int32<typename T::underlying>(
              pos1 + (h1p * input_width + w1p) * channels);
          for (int i = 0; i < 4; i++) {
            i32 pos1_int[vec_width];
            float pos1_fp[vec_width];
            pos1_int_v[i].store(pos1_int);
            vec::convert(pos1_int, pos1_fp, vec_width);
            pos1_fp_v[i] = Vectorized<float>::loadu(pos1_fp, 8);
          }
          Vectorized<float> h0lambda_v(h0lambda);
          Vectorized<float> h1lambda_v(h1lambda);
          Vectorized<float> w0lambda_v(w0lambda);
          Vectorized<float> w1lambda_v(w1lambda);
          Vectorized<float> input_zero_point_v(input_zero_point);
          Vectorized<float> result =
              h0lambda_v * (w0lambda_v * pos1_fp_v[0] + w1lambda_v * pos1_fp_v[1]) +
              h1lambda_v * (w0lambda_v * pos1_fp_v[2] + w1lambda_v * pos1_fp_v[3]) -
              input_zero_point_v;
          float result_fp[vec_width];
          result.store(result_fp);
          native::quantize_vec<T>(
              inverse_scale,
              output_zero_point,
              result_fp,
              reinterpret_cast<T*>(pos2),
              vec_width);
          pos1 += vec_width;
          pos2 += vec_width;
        }
      }
    #endif
      return c;
        */
}



pub fn qupsample_bilinear2d_nhwc_kernel(
        output:        &mut Tensor,
        input:         &Tensor,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64,
        nbatch:        i64,
        channels:      i64,
        align_corners: bool,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(
          input.scalar_type(), "upsample_bilinear2d_nhwc", [&]() {
            auto* idata = static_cast<Scalar*>(input.data_ptr());
            auto* odata = static_cast<Scalar*>(output.data_ptr());
            float inverse_scale = output.q_scale() / input.q_scale();
            const auto rheight = area_pixel_compute_scale<float>(
                input_height, output_height, align_corners, scales_h);
            const auto rwidth = area_pixel_compute_scale<float>(
                input_width, output_width, align_corners, scales_w);

            for (i64 b = 0; b < nbatch; ++b) {
              auto* i_p = reinterpret_cast<typename Scalar::underlying*>(
                  idata + b * input_height * input_width * channels);
              auto* o_p = reinterpret_cast<typename Scalar::underlying*>(
                  odata + b * output_height * output_width * channels);

              for (i64 h2 = 0; h2 < output_height; ++h2) {
                const auto h1r = area_pixel_compute_source_index<float>(
                    rheight, h2, align_corners, /*cubic=*/false);

                const i64 h1 = h1r;
                const i64 h1p = (h1 < input_height - 1) ? 1 : 0;
                const float h1lambda = h1r - h1;
                const float h0lambda = static_cast<float>(1.) - h1lambda;

                for (i64 w2 = 0; w2 < output_width; ++w2) {
                  const auto w1r = area_pixel_compute_source_index<float>(
                      rwidth, w2, align_corners, /*cubic=*/false);
                  const i64 w1 = w1r;
                  const i64 w1p = (w1 < input_width - 1) ? 1 : 0;

                  const float w1lambda = w1r - w1;
                  const float w0lambda = static_cast<float>(1.) - w1lambda;

                  i64 c = 0;
                  // We use float32 to do the computation
                  const typename Scalar::underlying* pos1 =
                      i_p + (h1 * input_width + w1) * channels;
                  typename Scalar::underlying* pos2 =
                      o_p + (h2 * output_width + w2) * channels;
                  // We have to isolate this function out because the VS does not
                  // expand the macro correctly.
                  c = do_quantized_bilinear_on_AVX2<Scalar>(
                      pos1,
                      pos2,
                      input_height,
                      input_width,
                      output_height,
                      output_width,
                      channels,
                      output.q_zero_point(),
                      input.q_zero_point(),
                      inverse_scale,
                      h0lambda,
                      h1lambda,
                      w0lambda,
                      w1lambda,
                      h1p,
                      w1p);
                  // 1) The following loop handles the remaining channels
                  // 2) It also handles the Non-AVX2 path
                  for (; c < channels; ++c) {
                    float result = h0lambda *
                            (w0lambda * pos1[0] + w1lambda * pos1[w1p * channels]) +
                        h1lambda *
                            (w0lambda * pos1[h1p * input_width * channels] +
                             w1lambda * pos1[(h1p * input_width + w1p) * channels]);
                    pos2[0] = native::quantize_val<Scalar>(
                                  inverse_scale,
                                  output.q_zero_point(),
                                  result - input.q_zero_point())
                                  .val_;
                    pos1 += 1;
                    pos2 += 1;
                  } // c
                } // w2
              } // h2
            } // b
          });
        */
}


pub fn qtopk_kernel(
        values:  &mut Tensor,
        indices: &mut Tensor,
        self_:   &Tensor,
        k:       i64,
        dim:     i64,
        largest: bool,
        sorted:  bool)  {
    
    todo!();
        /*
            auto sizes = self.sizes();
      auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .declare_static_shape(sizes, /*squash_dims=*/dim)
        .add_output(values)
        .add_output(indices)
        .add_input(self)
        .build();

      auto mode_values_stride = values.strides()[dim];
      auto mode_indices_stride = indices.strides()[dim];
      auto tmp_values_stride = self.strides()[dim];

      AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qtopk_cpu", [&] {
        auto loop = [&](char** data, const i64* strides, i64 n) {
          using underlying_t = typename Scalar::underlying;
          static_assert(sizeof(Scalar) == sizeof(underlying_t), "");
          return topk_impl_loop<underlying_t>(
              mode_values_stride, mode_indices_stride, tmp_values_stride,
              k, sizes[dim], largest, sorted, data, strides, n);
        };

        i64 grain_size = internal::GRAIN_SIZE / max(i64{1}, sizes[dim]);
        iter.for_each(loop, /*grain_size=*/grain_size);
      });
        */
}




#[inline] pub fn do_bn_compute<T>(
        x_ptr:               *mut T::underlying,
        y_ptr:               *mut T::underlying,
        fake_scale:          &mut Vectorized<f32>,
        in_zp_vec:           &mut Vectorized<f32>,
        scale_neg_zp_premul: &mut Vectorized<f32>,
        out_zero_point:      i64,
        out_zero_point_v:    &mut Vectorized<T>,
        alpha:               *mut f32,
        beta:                *mut f32,
        vec_num:             i64,
        relu_fused:          bool,
        k_vlen:              i64)  {

    todo!();
        /*
            using Vec = Vectorized<T>;
      auto vals_q = Vec::loadu(X_ptr);
      // Fake scale of 1.0 here, should not affect performance (FMA in place of sub)
      auto vals_dq = vals_q.dequantize(fake_scale, in_zp_vec, scale_neg_zp_premul);
      for (const auto idx : irange(vec_num)) {
        auto alpha_v = Vectorized<float>::loadu(alpha + idx * kVLen);
        auto beta_v = Vectorized<float>::loadu(beta + idx * kVLen);
        vals_dq[idx] = vec::fmadd(alpha_v, vals_dq[idx], beta_v);
      }
      auto outputs_q = Vec::quantize(vals_dq, /*output_scale=*/1.0f, out_zero_point, /*inv_output_scale=*/1.0f);
      // Fake scale again
      if (ReluFused) {
        outputs_q = outputs_q.maximum(out_zero_point_v);
      }
      outputs_q.store(Y_ptr, vec_num * kVLen);
        */
}


pub fn q_batch_norm_kernel<const ReluFused: bool>(
        N:              i64,
        C:              i64,
        hxw:            i64,
        in_zero_point:  i64,
        out_zero_point: i64,
        input:          &Tensor,
        a:              &Tensor,
        b:              &Tensor,
        output:         &mut Tensor)  {

    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qbatch_norm", [&]() {
        float* alpha = a.data_ptr<float>();
        float* beta = b.data_ptr<float>();
        auto minimum = numeric_limits<Scalar::underlying>::lowest();
        auto maximum = numeric_limits<Scalar::underlying>::max();
        Scalar::underlying* X =
            reinterpret_cast<Scalar::underlying*>(input.data_ptr());
        Scalar::underlying* Y = reinterpret_cast<Scalar::underlying*>(output.data_ptr());

        constexpr int kVLen = 8;
        const i64 outer_size = N * HxW;
        using Vec = Vectorized<Scalar>;
        // Hoisted variables
        auto in_zp_vec = Vectorized<float>(static_cast<float>(in_zero_point));
        auto fake_scale = Vectorized<float>(1.0f);
        auto scale_neg_zp_premul = fake_scale * in_zp_vec.neg();
        auto out_zero_point_v = Vec(Scalar(out_zero_point));
        const auto lanes = static_cast<i64>(Vec::float_num_vecs() * kVLen);
        for (const auto i : irange(outer_size)) {
          auto* X_ptr = reinterpret_cast<typename Scalar::underlying*>(X + i * C);
          auto* Y_ptr = reinterpret_cast<typename Scalar::underlying*>(Y + i * C);
          i64 ch = 0;

          for(; ch + lanes <= C; ch += lanes ) {
            do_bn_compute<Scalar>(
              X_ptr + ch,
              Y_ptr + ch,
              fake_scale,
              in_zp_vec,
              scale_neg_zp_premul,
              out_zero_point,
              out_zero_point_v,
              alpha + ch,
              beta + ch,
              Vec::float_num_vecs(),
              ReluFused,
              kVLen
            );
          }

          // for channel between 8 and 32, still use 32 width for performance
          // Benchmark shows it is faster than doing 8 channels each time
          i64 elem_size = C - ch;
          if ((lanes == 32) && elem_size >= kVLen) {
            i64 vec_num = elem_size / kVLen;
            vector<typename Scalar::underlying> buf_in(lanes);
            memcpy(buf_in.data(), X_ptr + ch, vec_num * kVLen); // 3 cycles
            do_bn_compute<Scalar>(
              buf_in.data(),
              Y_ptr + ch,
              fake_scale,
              in_zp_vec,
              scale_neg_zp_premul,
              out_zero_point,
              out_zero_point_v,
              alpha + ch,
              beta + ch,
              vec_num,
              ReluFused,
              kVLen
            );
            ch += vec_num * kVLen;
          }
          // for channels less than 8
          for (; ch < C; ++ch) {
            long quantized_down = out_zero_point +
                lrintf(alpha[ch] * (X_ptr[ch] - in_zero_point) +
                            beta[ch]);
            if (ReluFused) { // static if
              quantized_down = max<long>(quantized_down, out_zero_point);
            }
            Y_ptr[ch] = min<long>(
                max<long>(quantized_down, minimum), maximum);
          }
        }
    });
        */
}


pub fn fake_quantize_tensor_cachemask_kernel(
        output:    &mut Tensor,
        mask:      &mut Tensor,
        input:     &Tensor,
        sc:        f32,
        z_point:   i64,
        quant_min: i64,
        quant_max: i64)  {
    
    todo!();
        /*
            float inv_scale = 1.0f / sc;

      auto iter_combined = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(output)
        .add_output(mask)
        .add_input(input)
        .build();

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fake_quantize_tensor_cachemask_kernel_type_handling", [&] {
        iter_combined.for_each([&](char** data, const i64* strides, i64 n) {
          for (i64 i = 0; i < n; i++) {
            Scalar* output_val = (Scalar*)(data[0] + i * strides[0]);
            bool* mask_val = (bool*)(data[1] + i * strides[1]);
            Scalar* input_val = (Scalar*)(data[2] + i * strides[2]);

            const auto qval = static_cast<i64>(z_point + nearbyint(*input_val * inv_scale));
            *output_val = (fmin(fmax(qval, quant_min), quant_max) - z_point) * sc;
            *mask_val = ((quant_min <= qval) && (qval <= quant_max));
          }
        });
      });
        */
}


pub fn fake_quantize_learnable_tensor_grad_kernel_cpu(
        iter:        &mut TensorIterator,
        scale:       f32,
        inv_scale:   f32,
        zero_point:  i64,
        quant_min:   i64,
        quant_max:   i64,
        grad_factor: f32)  {
    
    todo!();
        /*
      float dscale_small = quant_min - zero_point;
      float dscale_big = quant_max - zero_point;
      iter.for_each([&](char** data, const i64* strides, i64 n) {
        /*  When a for_each call is made on a TensorIterator with multiple inputs and outputs,
            the order they are accessed follows the order they are built within the iterator.
            For example, if an iterator is built in the following order:
            auto iter = TensorIteratorConfig().
              .add_output(firstOutput)
              .add_output(secondOutput)
              .add_input(firstInput)
              .add_input(secondInput)
              .build()
            data will contain 4 pointers to pointers to values in the following order:
            firstOutput, secondOutput, firstInput, secondInput.
            Proper pointer referencing and dereferencing, along with the usage of strides
            (to move onto different elements), can allow accessing of the input and assignment
            to the right output.
        */
        for (i64 i = 0; i < n; i++) {
          float* dXOutput = (float*)(data[0] + i * strides[0]);
          float* dScaleOutput = (float*)(data[1] + i * strides[1]);
          float* dZeroPointOutput = (float*)(data[2] + i * strides[2]);
          float* XInput = (float*)(data[3] + i * strides[3]);
          float* dYInput = (float*)(data[4] + i * strides[4]);
          // Calculate gradients for X.
          i64 xqi = nearbyint(zero_point + (*XInput) * inv_scale);
          *dXOutput = (*dYInput) * (xqi >= quant_min && xqi <= quant_max);
          // Calculate gradients for scale and zero point.
          float xfqi = static_cast<float>((max(min(xqi, quant_max), quant_min) - zero_point) * scale);
          // Calculate gradients according to the gradient of the clamp function.
          if (xqi < quant_min || xqi > quant_max) {
            *dZeroPointOutput = (*dYInput) * (-1) * scale * grad_factor;
            *dScaleOutput = ((xqi < quant_min) ? ((*dYInput) * dscale_small) : ((*dYInput) * dscale_big)) * grad_factor;
          } else {
            *dZeroPointOutput = 0;
            *dScaleOutput = (*dYInput) * (xfqi - (*XInput)) * inv_scale * grad_factor;
          }
        }
      });
        */
}

pub fn fake_quant_per_channel_cachemask_cpu(
        iter:      &mut TensorIterator,
        iter_mask: &mut TensorIterator,
        quant_min: i64,
        quant_max: i64)  {
    
    todo!();
        /*
            // TODO(future, optional): read once, write twice.  Not done at the moment
      //   for simplicity, as we do not expect this to be a bottleneck.
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "fake_quantize_channel_cachemask_cpu_type_handling", [&] {
        // write mask
        cpu_kernel(iter_mask, [=](Scalar self, float scale, i64 zero_point) -> bool {
          float inv_scale = 1.0f / scale;
          const auto qval = static_cast<i64>(zero_point + nearbyint(self * inv_scale));
          return ((quant_min <= qval) && (qval <= quant_max));
        });

        // write fake_quant
        cpu_kernel(iter, [=](Scalar self, float scale, i64 zero_point) -> Scalar {
          float inv_scale = 1.0f / scale;
          return (fmin(
                      fmax(
                          static_cast<i64>(
                              zero_point + nearbyint(self * inv_scale)),
                          quant_min),
                      quant_max) -
                  zero_point) *
              scale;
        });
      });
        */
}



pub fn fake_quantize_learnable_channel_grad_kernel_cpu(
        iter:        &mut TensorIterator,
        quant_min:   i64,
        quant_max:   i64,
        grad_factor: f32)  {
    
    todo!();
        /*
            iter.for_each([&](char** data, const i64* strides, i64 n) {
        /*  To see how the input and outputs are referenced and assigned,
            please see the implemenetation of
            fake_quantize_learnable_tensor_grad_kernel_cpu.
        */
        for (i64 i = 0; i < n; i++) {
          float* dx_output = (float*)(data[0] + i * strides[0]);
          float* dscale_output = (float*)(data[1] + i * strides[1]);
          float* dzero_point_output = (float*)(data[2] + i * strides[2]);
          float* x_input = (float*)(data[3] + i * strides[3]);
          float* dy_input = (float*)(data[4] + i * strides[4]);
          float* scale_input = (float*)(data[5] + i * strides[5]);
          float* zero_point_input = (float*)(data[6] + i * strides[6]);

          float inv_scale = 1.0f / (*scale_input);
          float dscale_small = quant_min - (*zero_point_input);
          float dscale_big = quant_max - (*zero_point_input);
          // Calculate gradients for X.
          i64 xqi = nearbyint((*zero_point_input) + (*x_input) * inv_scale);
          *dx_output = (*dy_input) * (xqi >= quant_min && xqi <= quant_max);
          // Calculate gradients for scale and zero point.
          float xfqi = static_cast<float>((max(min(xqi, quant_max), quant_min) - (*zero_point_input)) * (*scale_input));
          if (xqi < quant_min || xqi > quant_max) {
            *dzero_point_output = (*dy_input) * (-1) * (*scale_input) * grad_factor;
            *dscale_output = ((xqi < quant_min) ? ((*dy_input) * dscale_small) : ((*dy_input) * dscale_big)) * grad_factor;
          } else {
            *dzero_point_output = 0;
            *dscale_output = (*dy_input) * (xfqi - (*x_input)) * inv_scale * grad_factor;
          }
        }
      });
        */
}

/**
  | Assumes X is composed of M groups of
  | N elements. Normalizes each of the groups and
  | optionally applies affine scaling. Useful for
  | LayerNorm, GroupNorm, InstanceNorm.
  */
pub fn quantized_normalize_kernel(

    // input tensor
    X:                  &Tensor,

    // weight (optional)
    gamma:              &Tensor,

    // bias (optional)
    beta:               &Tensor,

    // scaling applied elementwise if false, per channel if true
    affine_per_channel: bool,

    // only used if affine_per_channel is set
    num_channels:       i32,

    // only used if affine_per_channel is set
    num_groups:         i32,

    // number of groups
    M:                  i64,

    // number of elements in each group
    N:                  i64,

    eps:                f64,
    Y:                  *mut Tensor)  {

    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(X.scalar_type(), "quantized_layer_norm_kernel_impl_cpu", [&]() {
        using qVec = vec::Vectorized<Scalar>;
        using fVec = vec::Vectorized<float>;

        TORCH_INTERNAL_ASSERT(X.numel() == M * N, "Unexpected num elements in X");
        TORCH_INTERNAL_ASSERT(
            !gamma.defined() ||
            (!affine_per_channel && gamma.numel() == N) ||
            (affine_per_channel && gamma.numel() == num_channels),
            "Unexpected size of gamma");
        TORCH_INTERNAL_ASSERT(
            !beta.defined() ||
            (!affine_per_channel && beta.numel() == N) ||
            (affine_per_channel && beta.numel() == num_channels),
            "Unexpected size of beta");

        Scalar* X_data = X.data_ptr<Scalar>();
        const float* gamma_data = gamma.defined() ? gamma.data_ptr<float>() : nullptr;
        const float* beta_data = beta.defined() ? beta.data_ptr<float>() : nullptr;
        Scalar* Y_data = Y->data_ptr<Scalar>();
        const bool gamma_null = gamma_data == nullptr;
        const bool beta_null = beta_data == nullptr;
        i64 x_zp = X.q_zero_point();
        float x_scale = X.q_scale();
        fVec x_zp_vec((float)x_zp);
        fVec one_vec(1.0f);
        fVec zero_vec(0.0f);
        float x_fake_scale = 1.0f;
        fVec x_fake_scale_vec(x_fake_scale);
        fVec x_fake_scale_zp_neg_premul_vec = x_fake_scale_vec * x_zp_vec.neg();
        i64 y_zp = Y->q_zero_point();
        float y_scale = Y->q_scale();
        float y_inv_scale = 1.0f / y_scale;

        constexpr int kFloatVLen = 8;
        i64 kIntVLen = kFloatVLen * qVec::float_num_vecs();
        i64 kNumIntVecInLayer = N / kIntVLen;
        i64 kNonVecRemInLayer = N % kIntVLen;
        int channels_per_group = num_channels / num_groups;
        i64 NPerChannel = N / channels_per_group;
        i64 kNumIntVecInChannel = NPerChannel / kIntVLen;
        i64 kNonVecRemInChannel = NPerChannel % kIntVLen;

        parallel_for(0, M, 1, [&](i64 start, i64 end) {
          for (i64 i = start; i < end; ++i) {

            Scalar* X_ptr = X_data + i * N;
            Scalar* Y_ptr = Y_data + i * N;

            // First pass: calculate mean and variance.

            Scalar::underlying* X_ptr_underlying = reinterpret_cast<Scalar::underlying*>(X_ptr);
            auto l_sum_shifted = hsum(X_ptr_underlying, N);
            auto l_sum_sq_shifted = hsum_sq(X_ptr_underlying, N);
            float l_mean_shifted_div_scale_x = static_cast<float>(l_sum_shifted) / N;
            // mean(dqX) / scale_x
            float layer_mean_div_scale_x = l_mean_shifted_div_scale_x - x_zp;
            // var(dqX) / scale_x^2
            float layer_var_div_scale_x_sq =
              max(static_cast<float>(l_sum_sq_shifted) / N -
                  l_mean_shifted_div_scale_x * l_mean_shifted_div_scale_x, 0.0f);
            // scale_x / sqrt(var(dqX) + eps)
            float scale_x_div_layer_std = x_scale /
              sqrt(layer_var_div_scale_x_sq * x_scale * x_scale + eps);
            fVec layer_mean_div_scale_xVec(layer_mean_div_scale_x);
            fVec scale_x_div_layer_stdVec(scale_x_div_layer_std);

            // Second pass: normalize

            // TODO replace with TensorIterator implementation once #33166 is fixed.
            if (affine_per_channel) {

              // if scaling per channel, scaling parameters can be pre-multiplied
              // with normalization parameters
              for (i64 chIdx = 0; chIdx < channels_per_group; chIdx++) {
                int scalingIdx = (i * channels_per_group + chIdx) % (num_channels);
                float gamma = gamma_null ? 1.0f : gamma_data[scalingIdx];
                // scale_x / layer_std * gamma
                float gamma_p = scale_x_div_layer_std * gamma;
                float beta = beta_null ? 0.0f : beta_data[scalingIdx];
                fVec gamma_p_vec(gamma_p);
                fVec beta_vec(beta);

                i64 chStartIdx = chIdx * NPerChannel;
                i64 chEndIdx = chStartIdx + NPerChannel;

                for (i64 vecIdx = 0; vecIdx < kNumIntVecInChannel; vecIdx++) {
                  i64 vecStartIdx = chStartIdx + vecIdx * kIntVLen;
                  auto qXVec = qVec::loadu(X_ptr + vecStartIdx);
                  auto dqXVec = qXVec.dequantize(x_fake_scale_vec, x_zp_vec,
                      x_fake_scale_zp_neg_premul_vec);
                  for (auto &dq : dqXVec) {
                    dq =
                      (dq - layer_mean_div_scale_xVec) *
                        gamma_p_vec + beta_vec;
                    qVec::quantize(dqXVec, y_scale, y_zp, y_inv_scale)
                      .store(Y_ptr + vecStartIdx);
                  }
                }
                for (i64 remIdx = chEndIdx - kNonVecRemInChannel;
                     remIdx < chEndIdx;
                     remIdx++) {
                  auto qXVal = X_ptr[remIdx];
                  float dqXVal = native::dequantize_val(x_fake_scale, x_zp, qXVal);
                  float dqY =
                    (dqXVal - layer_mean_div_scale_x) * gamma_p + beta;
                  Y_ptr[remIdx] = native::quantize_val<Scalar>(y_scale, y_zp, dqY);
                }
              } // chIdx

            } else {

              for (i64 vecIdx = 0; vecIdx < kNumIntVecInLayer; vecIdx++) {
                i64 vecStartIdx = vecIdx * kIntVLen;
                auto qXVec = qVec::loadu(X_ptr + vecStartIdx);
                auto dqXVec = qXVec.dequantize(x_fake_scale_vec, x_zp_vec,
                    x_fake_scale_zp_neg_premul_vec);
                for (const auto dqXVecIdx : irange(dqXVec.size())) {
                  i64 vecVecStartIdx = vecStartIdx + dqXVecIdx * kFloatVLen;
                  auto gammaVec = gamma_null
                    ? one_vec
                    : fVec::loadu(gamma_data + vecVecStartIdx);
                  auto betaVec = beta_null
                    ? zero_vec
                    : fVec::loadu(beta_data + vecVecStartIdx);
                  dqXVec[dqXVecIdx] =
                    (dqXVec[dqXVecIdx] - layer_mean_div_scale_xVec) *
                      scale_x_div_layer_stdVec * gammaVec + betaVec;
                  qVec::quantize(dqXVec, y_scale, y_zp, y_inv_scale)
                    .store(Y_ptr + vecStartIdx);
                }
              }
              for (i64 remIdx = N - kNonVecRemInLayer; remIdx < N; remIdx++) {
                const float gamma_v = gamma_null ? 1.0f : gamma_data[remIdx];
                const float beta_v = beta_null ? 0.0f : beta_data[remIdx];
                auto qXVal = X_ptr[remIdx];
                float dqXVal = native::dequantize_val(x_fake_scale, x_zp, qXVal);
                float dqY =
                  ((dqXVal - layer_mean_div_scale_x) * scale_x_div_layer_std) * gamma_v + beta_v;
                Y_ptr[remIdx] = native::quantize_val<Scalar>(y_scale, y_zp, dqY);
              }
            }
          }
        }); // parallel_for

      });
        */
}


#[cfg(feature = "fbgemm")]
pub fn quantize_tensor_per_tensor_affine_cpu(
        rtensor:    &Tensor,
        qtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(
          qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
            check_tensor_memory_format(rtensor, qtensor);
            const float* rd = rtensor.data_ptr<float>();
            auto qd = reinterpret_cast<underlying_t*>(qtensor.data_ptr<Scalar>());
            fbgemm::TensorQuantizationParams qparams;
            qparams.scale = scale;
            qparams.zero_point = zero_point;
            qparams.precision = CHAR_BIT * sizeof(underlying_t);
            int num_tasks = get_num_threads();
            parallel_for(0, num_tasks, 1, [&](i64 begin, i64 end) {
              for (int task_id = begin; task_id < end; ++task_id) {
                fbgemm::Quantize<underlying_t, false /*LEGACY*/>(
                    rd, /*src=*/
                    qd, /*dst=*/
                    rtensor.numel(), /*len*/
                    qparams, /*qparams=*/
                    task_id, /*thread_id*/
                    num_tasks /*num_threads*/);
              }
            });
          });
        */
}

#[cfg(feature = "fbgemm")]
pub fn dequantize_tensor_per_tensor_affine_cpu(
        qtensor:    &Tensor,
        rtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(
          qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
            check_tensor_memory_format(qtensor, rtensor);
            const auto* qd =
                reinterpret_cast<const underlying_t*>(qtensor.data_ptr<Scalar>());
            fbgemm::TensorQuantizationParams qparams;
            qparams.scale = scale;
            qparams.zero_point = zero_point;
            qparams.precision = CHAR_BIT * sizeof(underlying_t);
            float* rd = rtensor.data_ptr<float>();
            int num_tasks = get_num_threads();
            parallel_for(0, num_tasks, 1, [&](i64 begin, i64 end) {
              for (int task_id = begin; task_id < end; ++task_id) {
                fbgemm::Dequantize<underlying_t>(
                    qd, /*src=*/
                    rd, /*dst=*/
                    qtensor.numel(), /*len=*/
                    qparams, /*qparams=*/
                    task_id, /*thread_id*/
                    num_tasks /*num_threads*/);
              }
            });
          });
        */
}

/**
  | Generic template defaults to naive
  | quantize implementation
  |
  */
#[cfg(not(feature = "fbgemm"))]
#[cfg(any(__ARM_NEON__,__aarch64__))]
pub fn quantize_tensor_arm<T>(
        in_:        *const f32,
        qtensor:    &mut Tensor,
        N:          i64,
        scale:      f32,
        zero_point: i32)  {

    todo!();
        /*
            auto out = qtensor.data_ptr<T>();
      for (int i = 0; i < N; ++i) {
        out[i] = native::quantize_val<T>(scale, zero_point, in[i]);
      }
        */
}

/**
  | Specialized implementation from Int8Quantize.
  |
  | There may be slight accuracy difference between
  | this and implementation of quantize_val
  |
  | TODO Update quantize_tensor_arm implementation
  | to follow quantize_val,
  |
  | i.e. f = Round(value/scale + zero_point)
  |
  | TODO Make quantize_tensor_arm work for other
  | datatypes too (int8, int32).
  */
#[cfg(not(feature = "fbgemm"))]
#[cfg(any(__ARM_NEON__,__aarch64__))]
pub fn quantize_tensor_arm_quint8(
        in_:        *const f32,
        qtensor:    &mut Tensor,
        N:          i64,
        scale:      f32,
        zero_point: i32)  {
    
    todo!();
        /*
            const float inv_scale = 1.0f / scale;
      u32 i = 0;
      auto out = (u8*)qtensor.data_ptr<quint8>();
      const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);
    #if defined(__ARM_NEON__)
      // magic float and magic int to take care of rounding
      // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
      // Some detail:
      // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
      // add a small number to a large number, the result rounds to the precision of
      // the least significant bit of the large number. For IEEE-754
      // single-precision number mantissa has 23 bits, and adding 2**23 would cause
      // rounding to the nearest even integer. The we cast to int and subtract the
      // same number (0x4B400000 is the integer representation of 12582912.0f) to
      // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
      // sign for negative numbers.
      const int32x4_t voffset = vdupq_n_s32(zero_point - 0x4B400000);
      const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
      for (i = 0; i + 8 < N; i += 8) {
        const float32x4_t vin0123 = vld1q_f32(in);
        in += 4;
        const float32x4_t vin4567 = vld1q_f32(in);
        in += 4;
        const int32x4_t vraw0123 = vaddq_s32(
            voffset,
            vreinterpretq_s32_f32(
                vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
        const int32x4_t vraw4567 = vaddq_s32(
            voffset,
            vreinterpretq_s32_f32(
                vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
        const int16x8_t vraw01234567 =
            vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
        const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
        vst1_u8(out, vout01234567);
        out += 8;
      }
      for (; i < N; ++i) {
        (*out++) = native::quantize_val_arm(scale, zero_point, (*in++));
      }
    #else
      const int16x8_t vzero_point = vdupq_n_s16((i16)(u16)zero_point);
      for (i = 0; i + 8 < N; i += 8) {
        const float32x4_t vin0123 = vld1q_f32(in);
        in += 4;
        const float32x4_t vin4567 = vld1q_f32(in);
        in += 4;
        const int32x4_t v0123_rounded = vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
        const int32x4_t v4567_rounded = vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
        const int16x8_t v01234567_packed = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded), vzero_point);
        const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
        vst1_u8(out, vout01234567);
        out += 8;
      }
      for (; i < N; ++i) {
        (*out++) = native::quantize_val_arm(scale, zero_point, (*in++));
      }
    #endif
        */
}

#[cfg(not(feature = "fbgemm"))]
pub fn quantize_tensor_per_tensor_affine_cpu(
        rtensor:    &Tensor,
        qtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64)  {
    
    todo!();
        /*
            #if defined(__ARM_NEON__) || defined(__aarch64__)
      AT_DISPATCH_QINT_TYPES(
          qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
            check_tensor_memory_format(rtensor, qtensor);
            const float* const rdata = rtensor.data_ptr<float>();
            quantize_tensor_arm<Scalar>(
                rdata, qtensor, rtensor.numel(), scale, zero_point);
          });
    #else
      // Fallback path
      AT_DISPATCH_QINT_TYPES(
          qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_cpu", [&]() {
            check_tensor_memory_format(rtensor, qtensor);
            const float* const rdata = rtensor.data_ptr<float>();
            auto qdata = qtensor.data_ptr<Scalar>();
            auto numel = rtensor.numel();
            for (int i = 0; i < numel; ++i) {
              qdata[i] = quantize_val<Scalar>(scale, zero_point, rdata[i]);
            }
          });
    #endif // __ARM_NEON__
        */
}

#[cfg(not(feature = "fbgemm"))]
pub fn dequantize_tensor_per_tensor_affine_cpu(
        qtensor:    &Tensor,
        rtensor:    &mut Tensor,
        scale:      f64,
        zero_point: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(
          qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_cpu", [&]() {
          check_tensor_memory_format(qtensor, rtensor);
            const auto* qd = qtensor.data_ptr<Scalar>();
            float* rd = rtensor.data_ptr<float>();
            auto numel = qtensor.numel();
            for (auto i = 0; i < numel; ++i) {
              rd[i] = dequantize_val<Scalar>(scale, zero_point, qd[i]);
            }
          });
        */
}

/**
  | TODO: add fbgemm for per channel
  |
  | Generic template defaults to naive quantize
  | implementation
  */
pub fn quantize_tensor_per_channel_impl<T>(
    rtensor:     &Tensor,
    qtensor:     &mut Tensor,
    scales:      &Tensor,
    zero_points: &Tensor,
    axis:        i64)  {

    todo!();
        /*
            // TODO: channels last kernel can be made faster.
      // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
      // For channels_last/3d however axis == 0 or 1.
      // Since current implemntation on channels_last format does not
      // cover per channel quant with arbitrary axis value, it is better
      // to check and fail.
      i64 batches = size_to_dim_(axis, rtensor.sizes());
      i64 elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
      i64 channels = rtensor.size(axis);
      auto scales_data = scales.data_ptr<double>();
      auto zero_points_data = zero_points.data_ptr<i64>();
      const float* in = rtensor.data_ptr<float>();
      auto out = qtensor.data_ptr<T>();
      if (axis == 1 &&
          (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
           rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
        // This code handles per channel quant when axis = 1 and
        // channels_last contig.
        // If axis = 0 and channels_last contig, implementation for channels
        // first (NCHW) works.
        for (auto b = 0; b < batches; ++b) {
          for (auto e = 0; e < elements_per_channel; ++e) {
            for (auto c = 0; c < channels; ++c) {
              auto i = b * channels * elements_per_channel + e * channels + c;
              out[i] = native::quantize_val<T>(
                  scales_data[c], zero_points_data[c], in[i]);
            }
          }
        }
      } else {
        for (auto b = 0; b < batches; ++b) {
          for (auto c = 0; c < channels; ++c) {
            for (auto e = 0; e < elements_per_channel; ++e) {
              auto i = b * channels * elements_per_channel +
                  c * elements_per_channel + e;
              out[i] = native::quantize_val<T>(
                  scales_data[c], zero_points_data[c], in[i]);
            }
          }
        }
      }
        */
}

/**
  | Specialized implementation from Int8Quantize.
  |
  | There may be slight accuracy difference between
  | this and implementation of quantize_val
  |
  | TODO Update quantize_tensor_per_channel_impl
  | implementation to follow quantize_val,
  | i.e. f = Round(value/scale + zero_point)
  |
  | TODO Make quantize_tensor_per_channel_impl work
  | for other datatypes too (int8, int32).
  */
#[cfg(any(__ARM_NEON__,__aarch64__))]
pub fn quantize_tensor_per_channel_impl_quint8(
    rtensor:     &Tensor,
    qtensor:     &mut Tensor,
    scales:      &Tensor,
    zero_points: &Tensor,
    axis:        i64)  {

    todo!();
        /*
            i64 batches = size_to_dim_(axis, rtensor.sizes());
      i64 elements_per_channel = size_from_dim_(axis + 1, rtensor.sizes());
      i64 channels = rtensor.size(axis);
      auto scales_data = scales.data_ptr<double>();
      auto zero_points_data = zero_points.data_ptr<i64>();
      const float* in = rtensor.data_ptr<float>();
      auto out = (u8*)qtensor.data_ptr<quint8>();
    #if defined(__ARM_NEON__)
      // magic float and magic int to take care of rounding
      // int magic_round(float f): interpret_int32(f + 12582912.0f) - 0x4B400000
      // Some detail:
      // 12582912.0f is 2**23 + 2**22. The trick is based on the fact that when you
      // add a small number to a large number, the result rounds to the precision of
      // the least significant bit of the large number. For IEEE-754
      // single-precision number mantissa has 23 bits, and adding 2**23 would cause
      // rounding to the nearest even integer. The we cast to int and subtract the
      // same number (0x4B400000 is the integer representation of 12582912.0f) to
      // get only the mantissa. This works if -2**22 < x < 2**22, but preserves the
      // sign for negative numbers.
      const float32x4_t vmagic_float = vdupq_n_f32(12582912.0f);
      // Copy reciprocal of scales (double) into float array
      // Copy zero_points with magic int (i64) into i32 array
      vector<float> inv_scales(channels);
      vector<i32> zero_points_int32t(channels);
      for (int i = 0; i < channels; ++i) {
        inv_scales[i] = 1.0f / (float)scales_data[i];
        zero_points_int32t[i] = (i32)(u32)zero_points_data[i] - 0x4B400000;
      }
      if (axis == 1 &&
          (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
           rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
        // This code handles per channel quant when axis = 1 and
        // channels_last contig.
        // If axis = 0 and channels_last contig, implementation for channels
        // first (NCHW) works.
        for (u32 b = 0; b < batches; ++b) {
          for (u32 e = 0; e < elements_per_channel; ++e) {
            u32 c = 0;
            while (c + 8 < channels) {
              const int32x4_t voffset0123 = vld1q_s32(&zero_points_int32t[c]);
              const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]);
              c += 4;
              const int32x4_t voffset4567 = vld1q_s32(&zero_points_int32t[c]);
              const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]);
              c += 4;
              const float32x4_t vin0123 = vld1q_f32(in);
              in += 4;
              const float32x4_t vin4567 = vld1q_f32(in);
              in += 4;
              const int32x4_t vraw0123 = vaddq_s32(
                  voffset0123,
                  vreinterpretq_s32_f32(
                      vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale0123))));
              const int32x4_t vraw4567 = vaddq_s32(
                  voffset4567,
                  vreinterpretq_s32_f32(
                      vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale4567))));
              const int16x8_t vraw01234567 =
                  vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
              const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
              vst1_u8(out, vout01234567);
              out += 8;
            }
            for (; c < channels; ++c) {
              (*out++) =
                  native::quantize_val_arm(scales_data[c], zero_points_data[c], (*in++));
            }
          }
        }
      } else {
        for (u32 b = 0; b < batches; ++b) {
          for (u32 c = 0; c < channels; ++c) {
            u32 e = 0;
            const int32x4_t voffset = vdupq_n_s32(zero_points_int32t[c]);
            const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
            for (; e + 8 < elements_per_channel; e += 8) {
              const float32x4_t vin0123 = vld1q_f32(in);
              in += 4;
              const float32x4_t vin4567 = vld1q_f32(in);
              in += 4;
              const int32x4_t vraw0123 = vaddq_s32(
                  voffset,
                  vreinterpretq_s32_f32(
                      vaddq_f32(vmagic_float, vmulq_f32(vin0123, vinv_scale))));
              const int32x4_t vraw4567 = vaddq_s32(
                  voffset,
                  vreinterpretq_s32_f32(
                      vaddq_f32(vmagic_float, vmulq_f32(vin4567, vinv_scale))));
              const int16x8_t vraw01234567 =
                  vcombine_s16(vqmovn_s32(vraw0123), vqmovn_s32(vraw4567));
              const uint8x8_t vout01234567 = vqmovun_s16(vraw01234567);
              vst1_u8(out, vout01234567);
              out += 8;
            }
            for (; e < elements_per_channel; ++e) {
              (*out++) =
                  native::quantize_val_arm(scales_data[c], zero_points_data[c], (*in++));
            }
          }
        }
      }
    #else // defined(__ARM_NEON__)
      // Copy scales (double) into float array
      // Copy zero_points (i64) into i16 array
      vector<float> inv_scales(channels);
      vector<i16> zero_points_int16t(channels);
      for (int i = 0; i < channels; ++i) {
        inv_scales[i] = 1.0f / (float)scales_data[i];
        zero_points_int16t[i] = (i16)(u16)zero_points_data[i];
      }
      if (axis == 1 &&
          (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
           rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
        // This code handles per channel quant when axis = 1 and
        // channels_last contig.
        // If axis = 0 and channels_last contig, implementation for channels
        // first (NCHW) works.
        for (u32 b = 0; b < batches; ++b) {
          for (u32 e = 0; e < elements_per_channel; ++e) {
            u32 c = 0;
            while (c + 8 < channels) {
              const int16x8_t vzero_point = vld1q_s16(&zero_points_int16t[c]);
              const float32x4_t vinv_scale0123 = vld1q_f32(&inv_scales[c]);
              c += 4;
              const float32x4_t vinv_scale4567 = vld1q_f32(&inv_scales[c]);
              c += 4;
              const float32x4_t vin0123 = vld1q_f32(in);
              in += 4;
              const float32x4_t vin4567 = vld1q_f32(in);
              in += 4;
              const int32x4_t v0123_rounded =
                  vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale0123));
              const int32x4_t v4567_rounded =
                  vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale4567));
              const int16x8_t v01234567_packed = vqaddq_s16(
                  vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
                  vzero_point);
              const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
              vst1_u8(out, vout01234567);
              out += 8;
            }
            for (; c < channels; ++c) {
              (*out++) =
                  native::quantize_val_arm(scales_data[c], zero_points_data[c], (*in++));
            }
          }
        }
      } else {
        for (u32 b = 0; b < batches; ++b) {
          for (u32 c = 0; c < channels; ++c) {
            u32 e = 0;
            const int16x8_t vzero_point = vdupq_n_s16(zero_points_int16t[c]);
            const float32x4_t vinv_scale = vdupq_n_f32(inv_scales[c]);
            for (; e + 8 < elements_per_channel; e += 8) {
              const float32x4_t vin0123 = vld1q_f32(in);
              in += 4;
              const float32x4_t vin4567 = vld1q_f32(in);
              in += 4;
              const int32x4_t v0123_rounded =
                  vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
              const int32x4_t v4567_rounded =
                  vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
              const int16x8_t v01234567_packed = vqaddq_s16(
                  vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
                  vzero_point);
              const uint8x8_t vout01234567 = vqmovun_s16(v01234567_packed);
              vst1_u8(out, vout01234567);
              out += 8;
            }
            for (; e < elements_per_channel; ++e) {
              (*out++) =
                  native::quantize_val_arm(scales_data[c], zero_points_data[c], (*in++));
            }
          }
        }
      }
    #endif // defined(__ARM_NEON__)
        */
}

pub fn quantize_tensor_per_channel_affine_cpu(
    rtensor:     &Tensor,
    qtensor:     &mut Tensor,
    scales:      &Tensor,
    zero_points: &Tensor,
    axis:        i64)  {

    todo!();
        /*
            TORCH_CHECK(
          rtensor.is_contiguous() || (axis <= 1),
          "If tensor is channels_last contig then per channel quantization "
          "is supported only for axis = 0 or 1.");
      AT_DISPATCH_QINT_TYPES(
          qtensor.scalar_type(), "quantize_tensor_per_channel_affine_cpu", [&]() {
            check_tensor_memory_format(rtensor, qtensor);
            quantize_tensor_per_channel_impl<Scalar>(
                rtensor, qtensor, scales, zero_points, axis);
          });
        */
}

pub fn dequantize_per_channel_affine_kernel<T, N, Q>(
    qtensor:     &Tensor,
    rtensor:     &mut Tensor,
    scales:      &Tensor,
    zero_points: &Tensor,
    axis:        i64,
    bit_width:   i32)  {

    let bit_width: i32 = bit_width.unwrap_or(8);

    todo!();
        /*
            // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
      // For channels_last/3d however axis == 0 or 1.
      // Since current implemntation on channels_last format does not
      // cover per channel quant with arbitrary axis value, it is better
      // to check and fail.
      TORCH_CHECK(rtensor.is_contiguous() || (axis <=1),
          "If tensor is channels_last contig then per channel quantization "
          "is supported only for axis = 0 or 1.");
      i64 batches = size_to_dim_(axis, rtensor.sizes());
      i64 elements_per_channel =
          size_from_dim_(axis + 1, rtensor.sizes());
      i64 channel = rtensor.size(axis);
      auto scales_data = scales.data_ptr<T>();
      auto zero_points_data = zero_points.data_ptr<N>();
      check_tensor_memory_format(qtensor, rtensor);
      const auto* qd = qtensor.data_ptr<Q>();
      float* rd = rtensor.data_ptr<float>();
      const auto elem_per_byte = 8 / bit_width;
      if (axis == 1 && (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
          rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
        for (auto b = 0; b < batches; ++b) {
          for (auto e = 0; e < elements_per_channel; ++e) {
            for (auto c = 0; c < channel; ++c) {
              auto i = b * channel * elements_per_channel + e * channel + c;
              // We need to convert the qint8 value to float to ensure the
              // subtraction subexpression returns a float
              auto qvalue = qd[i / elem_per_byte].val_;
              if (bit_width < 8) {
                qvalue >>= (i % elem_per_byte) * bit_width;
                qvalue &= (1 << bit_width) - 1;
              }
              rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
            }
          }
        }
      } else {
        for (auto b = 0; b < batches; ++b) {
          for (auto c = 0; c < channel; ++c) {
            for (auto e = 0; e < elements_per_channel; ++e) {
              auto i = b * channel * elements_per_channel +
                  c * elements_per_channel + e;
              // We need to convert the qint8 value to float to ensure the
              // subtraction subexpression returns a float
              auto qvalue = qd[i / elem_per_byte].val_;
              if (bit_width < 8) {
                qvalue >>= (i % elem_per_byte) * bit_width;
                qvalue &= (1 << bit_width) - 1;
              }
              rd[i] = (static_cast<float>(qvalue) - zero_points_data[c]) * scales_data[c];
            }
          }
        }
      }
        */
}



pub fn dequantize_tensor_per_channel_affine_cpu(
        qtensor:     &Tensor,
        rtensor:     &mut Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64)  {
    
    todo!();
        /*
            AT_DISPATCH_QINT_TYPES(
          qtensor.scalar_type(), "dequantize_tensor_per_channel_affine_cpu", [&]() {
            dequantize_per_channel_affine_kernel<double, i64, Scalar>(qtensor, rtensor, scales, zero_points, axis);
          });
        */
}

/// quantize stubs for floating point scale and
/// zero_point.
///
pub fn quantize_tensor_per_channel_float_qparams_cpu(
        rtensor:     &Tensor,
        qtensor:     &mut Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64)  {
    
    todo!();
        /*
            // For contiguous tensors, e.g. NCHW, arbitrary axis can be used.
      // For channels_last/3d however axis == 0 or 1.
      // Since current implemntation on channels_last format does not
      // cover per channel quant with arbitrary axis value, it is better
      // to check and fail.
      TORCH_CHECK(rtensor.is_contiguous() || (axis <=1),
          "If tensor is channels_last contig then per channel quantization "
          "is supported only for axis = 0 or 1.");
      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
          qtensor.scalar_type(), "quantize_tensor_per_channel_float_qparams_cpu", [&]() {
            i64 batches = size_to_dim_(axis, rtensor.sizes());
            i64 elements_per_channel =
                size_from_dim_(axis + 1, rtensor.sizes());
            i64 channel = rtensor.size(axis);
            auto scales_data = scales.data_ptr<float>();
            auto zero_points_data = zero_points.data_ptr<float>();
            check_tensor_memory_format(rtensor, qtensor);
            const float* rdata = rtensor.data_ptr<float>();
            auto qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<Scalar>());
            const auto elem_per_byte = CHAR_BIT / bit_width;
            int qvalue = 0;
            if (axis == 1 && (rtensor.is_contiguous(MemoryFormat::ChannelsLast) ||
                rtensor.is_contiguous(MemoryFormat::ChannelsLast3d))) {
              for (auto b = 0; b < batches; ++b) {
                for (auto e = 0; e < elements_per_channel; ++e) {
                  for (auto c = 0; c < channel; ++c) {
                    auto i = b * channel * elements_per_channel + e * channel + c;
                    qvalue = quantize_val_float_qparams(
                        scales_data[c], zero_points_data[c], rdata[i], quant_min, quant_max);
                    if (i % elem_per_byte == 0) {
                      qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
                    } else {
                      qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
                    }
                  }
                }
              }
            } else {
              for (auto b = 0; b < batches; ++b) {
                for (auto c = 0; c < channel; ++c) {
                  for (auto e = 0; e < elements_per_channel; ++e) {
                    auto i = b * channel * elements_per_channel +
                        c * elements_per_channel + e;
                    qvalue = quantize_val_float_qparams(
                        scales_data[c], zero_points_data[c], rdata[i], quant_min, quant_max);
                    if (i % elem_per_byte == 0) {
                      qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
                    } else {
                      qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
                    }
                  }
                }
              }
            }
          });
        */
}

pub fn dequantize_tensor_per_channel_float_qparams_cpu(
        qtensor:     &Tensor,
        rtensor:     &mut Tensor,
        scales:      &Tensor,
        zero_points: &Tensor,
        axis:        i64)  {
    
    todo!();
        /*
      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
          qtensor.scalar_type(), "dequantize_tensor_per_channel_float_qparams_cpu", [&]() {
            dequantize_per_channel_affine_kernel<float, float, Scalar>(qtensor, rtensor, scales, zero_points, axis, bit_width);
          });
        */
}


pub fn quantize_tensor_per_tensor_affine_sub_byte_cpu(
        rtensor:    &Tensor,
        qtensor:    &mut Tensor,
        scale:      f32,
        zero_point: f32)  {
    
    todo!();
        /*
            // TODO Use fbgemm kernel to pack values
      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
        qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
          check_tensor_memory_format(rtensor, qtensor);
          const float* const rdata = rtensor.data_ptr<float>();
          auto qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<Scalar>());
          auto numel = rtensor.numel();
          const auto elem_per_byte = CHAR_BIT / bit_width;
          for (int i = 0; i < numel; ++i) {
            float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
            i64 qvalue = lrintf(nearbyint(rdata[i] * inv_scale) + zero_point);
            qvalue = max(quant_min, min(qvalue, quant_max));

            // We pack sub_byte values and align them to a byte.
            // Eg. for 4-bits Index 0 is packed in the lower 4-bits
            // and index 1 is packed in the upper 4-bits.
            if (i % elem_per_byte == 0) {
              qdata[i / elem_per_byte] = static_cast<underlying_t>(qvalue);
            } else {
              qdata[i / elem_per_byte] |= static_cast<underlying_t>(qvalue << ((i % elem_per_byte) * bit_width));
            }
          } // for numel
        });
        */
}


pub fn dequantize_tensor_per_tensor_affine_sub_byte_cpu(
        qtensor:    &Tensor,
        rtensor:    &mut Tensor,
        scale:      f32,
        zero_point: f32)  {
    
    todo!();
        /*
            // TODO Use fbgemm kernel to pack values
      AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
        qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_sub_byte_cpu", [&]() {
          check_tensor_memory_format(rtensor, qtensor);
          auto rdata = rtensor.data_ptr<float>();
          const underlying_t* qdata = reinterpret_cast<underlying_t*>(qtensor.data_ptr<Scalar>());
          auto numel = rtensor.numel();
          const auto elem_per_byte = CHAR_BIT / bit_width;

          for (int i = 0; i < numel; ++i) {
            underlying_t qvalue = qdata[i / elem_per_byte];
            qvalue >>= (i % elem_per_byte) * bit_width;
            qvalue &= (1 << bit_width) - 1;
            rdata[i] = (static_cast<float>(qvalue) - zero_point) * scale;
          }
      });
        */
}

register_dispatch!{dequantize_tensor_per_channel_affine_stub         , &dequantize_tensor_per_channel_affine_cpu}
register_dispatch!{dequantize_tensor_per_tensor_affine_stub          , &dequantize_tensor_per_tensor_affine_cpu}
register_dispatch!{dequantize_tensor_per_channel_float_qparams_stub  , &dequantize_tensor_per_channel_float_qparams_cpu}
register_dispatch!{fake_quant_grad_learnable_tensor_stub             , &fake_quantize_learnable_tensor_grad_kernel_cpu}
register_dispatch!{fake_quant_per_channel_cachemask_stub             , &fake_quant_per_channel_cachemask_cpu}
register_dispatch!{fake_quant_tensor_cachemask_stub                  , &fake_quantize_tensor_cachemask_kernel}
register_dispatch!{qadaptive_avg_pool2d_nhwc_stub                    , &qadaptive_avg_pool2d_nhwc_kernel}
register_dispatch!{qadaptive_avg_pool3d_ndhwc_stub                   , &qadaptive_avg_pool3d_ndhwc_kernel}
register_dispatch!{qadd_relu_stub                                    , &qadd_kernel_fused}
register_dispatch!{qadd_scalar_relu_stub                             , &qadd_scalar_kernel_fused}
register_dispatch!{qadd_scalar_stub                                  , &qadd_scalar_kernel_unfused}
register_dispatch!{qadd_stub                                         , &qadd_kernel_unfused}
register_dispatch!{qavg_pool2d_nhwc_stub                             , &qavg_pool2d_nhwc_kernel}
register_dispatch!{qavg_pool3d_nhwc_stub                             , &qavg_pool3d_nhwc_kernel}
register_dispatch!{qbatch_norm_relu_stub                             , &q_batch_norm_kernel_fused}
register_dispatch!{qbatch_norm_stub                                  , &q_batch_norm_kernel_unfused}
register_dispatch!{qcat_nhwc_stub                                    , &qcat_nhwc_kernel_unfused}
register_dispatch!{qcat_relu_nhwc_stub                               , &qcat_nhwc_kernel_fused}
register_dispatch!{qclamp_stub                                       , &qclamp_kernel}
register_dispatch!{qclamp_min_stub                                   , &qclamp_min_kernel}
register_dispatch!{qclamp_max_stub                                   , &qclamp_max_kernel}
register_dispatch!{qelu_stub                                         , &qelu_kernel}
register_dispatch!{qhardsigmoid_stub                                 , &qhardsigmoid_kernel}
register_dispatch!{qhardswish_stub                                   , &qhardswish_kernel}
register_dispatch!{qmaxpool_2d_nhwc_stub                             , &qmaxpool_2d_nhwc_kernel}
register_dispatch!{qmul_relu_stub                                    , &qmul_kernel_fused}
register_dispatch!{qmul_stub                                         , &qmul_kernel_unfused}
register_dispatch!{qrelu6_stub                                       , &qrelu6_kernel}
register_dispatch!{qrelu_leaky_stub                                  , &leaky_qrelu_out_kernel}
register_dispatch!{qrelu_stub                                        , &qrelu_kernel}
register_dispatch!{qsigmoid_stub                                     , &qsigmoid_kernel}
register_dispatch!{qtanh_stub                                        , &qtanh_kernel}
register_dispatch!{qthreshold_stub                                   , &qthreshold_kernel}
register_dispatch!{qtopk_stub                                        , &qtopk_kernel}
register_dispatch!{fake_quant_grad_learnable_channel_stub            , &fake_quantize_learnable_channel_grad_kernel_cpu}
register_dispatch!{quantize_tensor_per_tensor_affine_stub            , &quantize_tensor_per_tensor_affine_cpu}
register_dispatch!{quantize_tensor_per_channel_affine_stub           , &quantize_tensor_per_channel_affine_cpu}
register_dispatch!{quantize_tensor_per_channel_float_qparams_stub    , &quantize_tensor_per_channel_float_qparams_cpu}
register_dispatch!{quantized_normalize_stub                          , &quantized_normalize_kernel}
register_dispatch!{qupsample_bilinear2d_nhwc_stub                    , &qupsample_bilinear2d_nhwc_kernel}
register_dispatch!{quantize_tensor_per_tensor_affine_sub_byte_stub   , &quantize_tensor_per_tensor_affine_sub_byte_cpu}
register_dispatch!{dequantize_tensor_per_tensor_affine_sub_byte_stub , &dequantize_tensor_per_tensor_affine_sub_byte_cpu}
