// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-microkernel-tester.h]

pub struct GemmMicrokernelTester {
    mr:           usize, // default = { 1 }
    nr:           usize, // default = { 1 }
    np:           usize, // default = { 1 }
    kr:           usize, // default = { 1 }
    m:            usize, // default = { 1 }
    n:            usize, // default = { 1 }
    k:            usize, // default = { 1 }
    ks:           usize, // default = { 1 }
    a_stride:     usize, // default = { 0 }
    c_stride:     usize, // default = { 0 }
    a_zero_point: u8, // default = { 127 }
    b_zero_point: u8, // default = { 127 }
    qmin:         u8, // default = { 0 }
    qmax:         u8, // default = { 255 }
    iterations:   usize, // default = { 15 }
    multiplier:   f32, // default = { 2.0f }
}

impl GemmMicrokernelTester {
    
    #[inline] pub fn mr(&mut self, mr: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->mr_ = mr;
        return *this;
        */
    }
    
    #[inline] pub fn mr(&self) -> usize {
        
        todo!();
        /*
            return this->mr_;
        */
    }
    
    #[inline] pub fn nr(&mut self, nr: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->nr_ = nr;
        return *this;
        */
    }
    
    #[inline] pub fn nr(&self) -> usize {
        
        todo!();
        /*
            return this->nr_;
        */
    }
    
    #[inline] pub fn np(&mut self, np: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->np_ = np;
        return *this;
        */
    }
    
    #[inline] pub fn np(&self) -> usize {
        
        todo!();
        /*
            return this->np_;
        */
    }
    
    #[inline] pub fn kr(&mut self, kr: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->kr_ = kr;
        return *this;
        */
    }
    
    #[inline] pub fn kr(&self) -> usize {
        
        todo!();
        /*
            return this->kr_;
        */
    }
    
    #[inline] pub fn m(&mut self, m: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->m_ = m;
        return *this;
        */
    }
    
    #[inline] pub fn m(&self) -> usize {
        
        todo!();
        /*
            return this->m_;
        */
    }
    
    #[inline] pub fn n(&mut self, n: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->n_ = n;
        return *this;
        */
    }
    
    #[inline] pub fn n(&self) -> usize {
        
        todo!();
        /*
            return this->n_;
        */
    }
    
    #[inline] pub fn k(&mut self, k: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->k_ = k;
        return *this;
        */
    }
    
    #[inline] pub fn k(&self) -> usize {
        
        todo!();
        /*
            return this->k_;
        */
    }
    
    #[inline] pub fn ks(&mut self, ks: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->ks_ = ks;
        return *this;
        */
    }
    
    #[inline] pub fn ks(&self) -> usize {
        
        todo!();
        /*
            return this->ks_;
        */
    }
    
    #[inline] pub fn packedk(&self) -> usize {
        
        todo!();
        /*
            return k() % kr() == 0 ? k() : (k() / kr() + 1) * kr();
        */
    }
    
    #[inline] pub fn packedn(&self) -> usize {
        
        todo!();
        /*
            return n() % np() == 0 ? n() : (n() / np() + 1) * np();
        */
    }
    
    #[inline] pub fn biasn(&self) -> usize {
        
        todo!();
        /*
            return n() % nr() == 0 ? n() : (n() / nr() + 1) * nr();
        */
    }
    
    #[inline] pub fn a_stride(&mut self, a_stride: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->aStride_ = aStride;
        return *this;
        */
    }
    
    #[inline] pub fn a_stride(&self) -> usize {
        
        todo!();
        /*
            return this->aStride_ == 0 ? k() : this->aStride_;
        */
    }
    
    #[inline] pub fn c_stride(&mut self, c_stride: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->cStride_ = cStride;
        return *this;
        */
    }
    
    #[inline] pub fn c_stride(&self) -> usize {
        
        todo!();
        /*
            return this->cStride_ == 0 ? n() : this->cStride_;
        */
    }
    
    #[inline] pub fn a_zero_point(&mut self, a_zero_point: u8) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->aZeroPoint_ = aZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn a_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->aZeroPoint_;
        */
    }
    
    #[inline] pub fn b_zero_point(&mut self, b_zero_point: u8) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->bZeroPoint_ = bZeroPoint;
        return *this;
        */
    }
    
    #[inline] pub fn b_zero_point(&self) -> u8 {
        
        todo!();
        /*
            return this->bZeroPoint_;
        */
    }
    
    #[inline] pub fn multiplier(&mut self, multiplier: f32) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->multiplier_ = multiplier;
        return *this;
        */
    }
    
    #[inline] pub fn multiplier(&self) -> f32 {
        
        todo!();
        /*
            return this->multiplier_;
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->qmin_ = qmin;
        return *this;
        */
    }
    
    #[inline] pub fn qmin(&self) -> u8 {
        
        todo!();
        /*
            return this->qmin_;
        */
    }
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->qmax_ = qmax;
        return *this;
        */
    }
    
    #[inline] pub fn qmax(&self) -> u8 {
        
        todo!();
        /*
            return this->qmax_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut GemmMicrokernelTester {
        
        todo!();
        /*
            this->iterations_ = iterations;
        return *this;
        */
    }
    
    #[inline] pub fn iterations(&self) -> usize {
        
        todo!();
        /*
            return this->iterations_;
        */
    }
    
    pub fn test(&self, qgemm: PyTorchQ8GemmUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());
        ASSERT_GE(k(), kr());

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);
        auto f32rng =
            bind(uniform_real_distribution<float>(1, 5), rng);

        vector<u8> a((m() - 1) * aStride() + k() + 8);
        vector<u8> b(n() * k());
        vector<i32> bias(n());
        vector<u8, AlignedAllocator<u8, 32>> packedW(
            packedN() * packedK() + biasN() * sizeof(u32) / sizeof(u8));
        vector<u8> c((m() - 1) * cStride() + n());
        vector<i32> acc(m() * n());
        vector<u8> cRef(m() * n());

        const u8* aPtr = a.data() + 8;

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(b.begin(), b.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          fill(c.begin(), c.end(), 0xA5);

          fill(packedW.begin(), packedW.end(), bZeroPoint());

          usize num_zero_points_padded = n() + 8;
          vector<u8> kernel_zero_points
            (num_zero_points_padded, bZeroPoint());
          generate(kernel_zero_points.begin(), kernel_zero_points.end(), ref(u8rng));
          pytorch_pack_q8gemm_w(
              n(),
              k(),
              nr(),
              np(),
              kr(),
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              aZeroPoint(),
              bZeroPoint(),
    #endif
              b.data(),
              bias.data(),
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              kernel_zero_points.data(),
    #endif
              packedW.data());

          ASSERT_NE(
              *max_element(a.cbegin(), a.cend()),
              *min_element(a.cbegin(), a.cend()));
          ASSERT_NE(
              *max_element(b.cbegin(), b.cend()),
              *min_element(b.cbegin(), b.cend()));

          /* Compute 32-bit results and output quantization arguments */
          fill(acc.begin(), acc.end(), 0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize kIndex = 0; kIndex < k(); kIndex++) {
                ASSERT_LE(n(), packedN());
                ASSERT_LT(mIndex * n() + nIndex, acc.size());
                ASSERT_LT(mIndex * k() + kIndex, a.size());
                acc[mIndex * n() + nIndex] +=
                    (i32(aPtr[mIndex * aStride() + kIndex]) -
                     i32(aZeroPoint())) *
                    (i32(b[nIndex * k() + kIndex]) - i32(kernel_zero_points[nIndex]));
              }
              acc[mIndex * n() + nIndex] += bias[nIndex];
            }
          }

          const i32 accMin = *min_element(acc.cbegin(), acc.cend());
          const i32 accMax = *max_element(acc.cbegin(), acc.cend());
          if (m() * n() >= 3) {
            ASSERT_NE(accMax, accMin)
                << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                << ", M x N x K = " << m() << " x " << n() << " x " << k();
          }

          const double cScale = u32(accMax - accMin) >= 256
              ? double(u32(accMax - accMin)) / 255.0
              : 1.00001;
          const u8 cZeroPoint = u8(max(
              min(
                  lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
                  long(u8::max)),
              long(u8::min)));

          vector<float> requantization_scales(num_zero_points_padded);
          auto scale_generator = [&]() -> float {return (f32rng()/cScale);};
          generate(
              requantization_scales.begin(),
              requantization_scales.end(),
              ref(scale_generator));
          const union pytorch_qnnp_conv_quantization_params quantizationParams =
              pytorch_qnnp_compute_conv_quantization_params(
                  aZeroPoint(),
                  kernel_zero_points.data(),
                  requantization_scales.data(),
                  cZeroPoint,
                  qmin(),
                  qmax());
          const union pytorch_qnnp_fp32_requantization_params
              scalarRequantizationParams =
                  pytorch_qnnp_compute_scalar_fp32_requantization_params(
                      requantization_scales.data(), cZeroPoint, qmin(), qmax());

          qgemm(
              m(),
              n(),
              k(),
              aPtr,
              aStride() * sizeof(u8),
              packedW.data(),
              c.data(),
              cStride() * sizeof(u8),
              0,
              &quantizationParams);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
    #if defined(__arm__) || defined(_M_ARM)
              cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize_magic(
                  acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);
    #else
              cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize(
                  acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);
    #endif
            }
          }

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_LE(u32(c[mIndex * cStride() + nIndex]), u32(qmax()));
              ASSERT_GE(u32(c[mIndex * cStride() + nIndex]), u32(qmin()));
              ASSERT_EQ(
                  u32(c[mIndex * cStride() + nIndex]),
                  u32(cRef[mIndex * n() + nIndex]))
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << (u32)cRef[mIndex * n() + nIndex]
                  << " (accumulator = " << acc[mIndex * n() + nIndex]
                  << "), optimized = " << (u32)c[mIndex * cStride() + nIndex]
                  << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k()
                  << ", requantization scale = " << requantization_scales[nIndex]
                  << ", output zero point = " << i32(cZeroPoint);
            }
          }
        }
        */
    }
    
    pub fn test(&self, qgemm: PyTorchQ8GemmDqUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());
        ASSERT_GE(k(), kr());

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> a((m() - 1) * aStride() + k() + 8);
        vector<u8> b(n() * k());
        vector<float, AlignedAllocator<float, 32>> bias(max<usize>(8, n()));
        vector<u8, AlignedAllocator<u8, 32>> packedW(
            packedN() * packedK() + biasN() * sizeof(u32) / sizeof(u8));
        vector<float> c((m() - 1) * cStride() + n());
        vector<float> acc(m() * n());

        const u8* aPtr = a.data() + 8;

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(b.begin(), b.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          fill(c.begin(), c.end(), 0.0f);

          fill(packedW.begin(), packedW.end(), bZeroPoint());

          usize num_zero_points_padded = n() + 8;
          vector<u8> kernel_zero_points
            (num_zero_points_padded, bZeroPoint());
          generate(kernel_zero_points.begin(), kernel_zero_points.end(), ref(u8rng));
          pytorch_pack_q8gemm_w(
              n(),
              k(),
              nr(),
              np(),
              kr(),
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              aZeroPoint(),
              bZeroPoint(),
    #endif
              b.data(),
              nullptr,
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              kernel_zero_points.data(),
    #endif
              packedW.data());

          ASSERT_NE(
              *max_element(a.cbegin(), a.cend()),
              *min_element(a.cbegin(), a.cend()));
          ASSERT_NE(
              *max_element(b.cbegin(), b.cend()),
              *min_element(b.cbegin(), b.cend()));

          auto f32rng =
              bind(uniform_real_distribution<float>(1, 5), rng);
          vector<float> dequantization_scales(num_zero_points_padded);
          generate(
              dequantization_scales.begin(),
              dequantization_scales.end(),
              ref(f32rng));
          /* Compute 32-bit results and output quantization arguments */
          fill(acc.begin(), acc.end(), 0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize kIndex = 0; kIndex < k(); kIndex++) {
                ASSERT_LE(n(), packedN());
                ASSERT_LT(mIndex * n() + nIndex, acc.size());
                ASSERT_LT(mIndex * k() + kIndex, a.size());
                acc[mIndex * n() + nIndex] +=
                    (i32(aPtr[mIndex * aStride() + kIndex]) -
                     i32(aZeroPoint())) *
                    (i32(b[nIndex * k() + kIndex]) - i32(kernel_zero_points[nIndex]));
              }
              acc[mIndex * n() + nIndex] =
                acc[mIndex * n() + nIndex] *
                dequantization_scales[nIndex] +
                bias[nIndex];
            }
          }

          const struct pytorch_qnnp_conv_dynamic_quantization_params quantizationParams{
            aZeroPoint(),
            kernel_zero_points.data(),
            dequantization_scales.data(),
          };

          qgemm(
              m(),
              n(),
              k(),
              aPtr,
              aStride() * sizeof(u8),
              packedW.data(),
              bias.data(),
              c.data(),
              cStride(),
              0,
              &quantizationParams);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_EQ(
                  c[mIndex * cStride() + nIndex],
                  acc[mIndex * n() + nIndex])
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << acc[mIndex * n() + nIndex]
                  << ", optimized = " << c[mIndex * cStride() + nIndex]
                  << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
        }
        */
    }
    
    pub fn test(&self, qconv: PyTorchQ8ConvUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());
        ASSERT_GE(k(), kr());

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);
        auto f32rng =
            bind(uniform_real_distribution<float>(1, 5), rng);

        vector<u8> a((mr() - 1) * aStride() + k() + 8);
        vector<u8> b(n() * ks() * k());
        vector<u8, AlignedAllocator<u8, 32>> packedW(
            ks() * packedN() * packedK() +
            biasN() * sizeof(u32) / sizeof(u8));
        vector<i32> bias(n());
        vector<u8> c((m() - 1) * cStride() + n());
        vector<i32> acc(m() * n());
        vector<u8> cRef(m() * n());
        vector<const u8*> im2col(mr() * ks());

        const u8* aPtr = a.data() + 8;

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(b.begin(), b.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          fill(c.begin(), c.end(), 0xA5);

          fill(packedW.begin(), packedW.end(), bZeroPoint());

          usize num_zero_points_padded = n() + 8;
          vector<u8> kernel_zero_points
            (num_zero_points_padded, bZeroPoint());
          generate(kernel_zero_points.begin(), kernel_zero_points.end(), ref(u8rng));

          pytorch_pack_q8conv_w(
              n(),
              ks(),
              k(),
              np(),
              kr(),
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              aZeroPoint(),
              bZeroPoint(),
    #endif
              b.data(),
              bias.data(),
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              kernel_zero_points.data(),
    #endif
              packedW.data());

          ASSERT_NE(
              *max_element(a.cbegin(), a.cend()),
              *min_element(a.cbegin(), a.cend()));
          ASSERT_NE(
              *max_element(b.cbegin(), b.cend()),
              *min_element(b.cbegin(), b.cend()));

          for (usize ksIndex = 0; ksIndex < ks(); ksIndex++) {
            for (usize mIndex = 0; mIndex < mr(); mIndex++) {
              im2col[ksIndex * mr() + mIndex] = aPtr + aStride() * mIndex;
            }
          }
          shuffle(im2col.begin(), im2col.end(), rng);
          for (usize ksIndex = 0; ksIndex < ks(); ksIndex++) {
            for (usize mIndex = m(); mIndex < mr(); mIndex++) {
              im2col[ksIndex * mr() + mIndex] = im2col[ksIndex * mr() + m() - 1];
            }
          }

          /* Compute 32-bit results and output quantization arguments */
          fill(acc.begin(), acc.end(), 0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize ksIndex = 0; ksIndex < ks(); ksIndex++) {
                for (usize kBlockStart = 0; kBlockStart < k();
                     kBlockStart += kr()) {
                  for (usize kBlockOffset = 0;
                       kBlockOffset < min(k() - kBlockStart, kr());
                       kBlockOffset++) {
                    ASSERT_LT(ksIndex * mr() + mIndex, im2col.size());
                    ASSERT_LT(kBlockStart + kBlockOffset, k());
                    ASSERT_LT(kBlockStart + kBlockOffset, aStride());

                    acc[mIndex * n() + nIndex] +=
                        (i32(im2col[ksIndex * mr() + mIndex]
                                       [kBlockStart + kBlockOffset]) -
                         i32(aZeroPoint())) *
                        (i32(
                             b[(nIndex * ks() + ksIndex) * k() + kBlockStart +
                               kBlockOffset]) -
                         i32(kernel_zero_points[nIndex]));
                  }
                }
              }
              acc[mIndex * n() + nIndex] += bias[nIndex];
            }
          }

          const i32 accMin = *min_element(acc.cbegin(), acc.cend());
          const i32 accMax = *max_element(acc.cbegin(), acc.cend());
          if (m() * n() >= 3) {
            ASSERT_NE(accMax, accMin)
                << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                << ", M x N x K = " << m() << " x " << n() << " x " << k();
          }

          const double cScale = u32(accMax - accMin) >= 256
              ? double(u32(accMax - accMin)) / 255.0
              : 1.00001;
          const u8 cZeroPoint = u8(max(
              min(
                  lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
                  long(u8::max)),
              long(u8::min)));

          vector<float> requantization_scales(num_zero_points_padded, 1.0f / float(cScale));
          auto scale_generator = [&]() -> float {return (f32rng()/cScale);};
          generate(
              requantization_scales.begin(),
              requantization_scales.end(),
              ref(scale_generator));
          const union pytorch_qnnp_conv_quantization_params quantizationParams =
              pytorch_qnnp_compute_conv_quantization_params(
                  aZeroPoint(),
                  kernel_zero_points.data(),
                  requantization_scales.data(),
                  cZeroPoint,
                  qmin(),
                  qmax());
          const union pytorch_qnnp_fp32_requantization_params
              scalarRequantizationParams =
                  pytorch_qnnp_compute_scalar_fp32_requantization_params(
                      requantization_scales.data(), cZeroPoint, qmin(), qmax());

          qconv(
              m(),
              n(),
              k(),
              ks(),
              im2col.data(),
              packedW.data(),
              c.data(),
              cStride() * sizeof(u8),
              0,
              &quantizationParams);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
    #if defined(__arm__) || defined(_M_ARM)
              cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize_magic(
                  acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);
    #else
              cRef[mIndex * n() + nIndex] = pytorch_qnnp_fp32_requantize(
                  acc[mIndex * n() + nIndex], scalarRequantizationParams, nIndex);
    #endif
            }
          }

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_LE(u32(c[mIndex * cStride() + nIndex]), u32(qmax()));
              ASSERT_GE(u32(c[mIndex * cStride() + nIndex]), u32(qmin()));
              ASSERT_EQ(
                  u32(c[mIndex * cStride() + nIndex]),
                  u32(cRef[mIndex * n() + nIndex]))
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << u32(cRef[mIndex * n() + nIndex])
                  << " (accumulator = " << acc[mIndex * n() + nIndex]
                  << "), optimized = " << u32(c[mIndex * cStride() + nIndex])
                  << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k()
                  << ", requantization scale = " << requantization_scales[nIndex]
                  << ", output zero point = " << i32(cZeroPoint);
            }
          }
        }
        */
    }
    
    pub fn q8gemm_compute_row_sum(
        a:          *const u8,
        m:          usize,
        k:          usize,
        stride:     usize,
        multiplier: i32,
        row_sum:    *mut i32,
        q8sum_rows: PyTorchQ8SumRowsUKernelFunction)  {
        
        todo!();
        /*
            const usize block_size = 4;
        for (usize block_start = 0; block_start < m; block_start += block_size) {
          q8sum_rows(
              a + block_start * stride,
              min(block_size, m - block_start),
              k,
              stride,
              multiplier,
              row_sum + block_start);
        }
        */
    }
    
    pub fn test(&self, qgemm: PyTorchQ8GemmXzpUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());
        ASSERT_GE(k(), kr());

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> a((m() - 1) * aStride() + k() + 8);
        vector<u8> b(n() * k());
        vector<i32> bias(n());
        vector<u8, AlignedAllocator<u8, 32>> packedW(
            packedN() * packedK() + biasN() * sizeof(u32) / sizeof(u8));
        vector<i32> aRowSums(m());
        vector<u8> c((m() - 1) * cStride() + n());
        vector<i32> acc(m() * n());
        vector<u8> cRef(m() * n());

        const u8* aPtr = a.data() + 8;

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(u8rng));
          generate(b.begin(), b.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));

          fill(packedW.begin(), packedW.end(), 0);
          pytorch_pack_swizzle_q8gemm_b(
              n(),
              k(),
              np(),
              kr(),
              8,
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
              aZeroPoint(),
              bZeroPoint(),
    #endif
              b.data(),
              bias.data(),
              packedW.data());

          ASSERT_NE(
              *max_element(a.cbegin(), a.cend()),
              *min_element(a.cbegin(), a.cend()));
          ASSERT_NE(
              *max_element(b.cbegin(), b.cend()),
              *min_element(b.cbegin(), b.cend()));

          fill(aRowSums.begin(), aRowSums.end(), 0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            i32 sum = 0;
            for (usize kIndex = 0; kIndex < k(); kIndex++) {
              sum += i32(aPtr[mIndex * aStride() + kIndex]);
            }
            aRowSums[mIndex] = -sum * i32(bZeroPoint());
          }

          /* Compute 32-bit results and output quantization arguments */
          fill(acc.begin(), acc.end(), 0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize kIndex = 0; kIndex < k(); kIndex++) {
                ASSERT_LE(n(), packedN());
                ASSERT_LT(mIndex * n() + nIndex, acc.size());
                ASSERT_LT(mIndex * k() + kIndex, a.size());
                acc[mIndex * n() + nIndex] +=
                    (i32(aPtr[mIndex * aStride() + kIndex]) -
                     i32(aZeroPoint())) *
                    (i32(b[nIndex * k() + kIndex]) - i32(bZeroPoint()));
              }
              acc[mIndex * n() + nIndex] += bias[nIndex];
            }
          }

          const i32 accMin = *min_element(acc.cbegin(), acc.cend());
          const i32 accMax = *max_element(acc.cbegin(), acc.cend());
          if (m() * n() >= 3) {
            ASSERT_NE(accMax, accMin)
                << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                << ", M x N x K = " << m() << " x " << n() << " x " << k();
          }

          const double cScale = u32(accMax - accMin) >= 256
              ? double(u32(accMax - accMin)) / 255.0
              : 1.00001;
          const u8 cZeroPoint = u8(max(
              min(
                  lrint(127.5 - 0.5 * double(accMin + accMax) / cScale),
                  long(u8::max)),
              long(u8::min)));

          const float requantizationScale = 1.0f / float(cScale);
          const union pytorch_qnnp_q31_requantization_params requantizationParams =
              pytorch_qnnp_compute_requantization_params(
                  requantizationScale, cZeroPoint, qmin(), qmax());
          const union pytorch_qnnp_q31_requantization_params
              scalarRequantizationParams =
                  pytorch_qnnp_compute_scalar_requantization_params(
                      requantizationScale, cZeroPoint, qmin(), qmax());

          fill(c.begin(), c.end(), 0xA5);
          qgemm(
              m(),
              n(),
              k(),
              aPtr,
              aStride(),
              aRowSums.data(),
              packedW.data(),
              c.data(),
              cStride(),
              &requantizationParams);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              cRef[mIndex * n() + nIndex] = pytorch_qnnp_q31_requantize(
                  acc[mIndex * n() + nIndex], scalarRequantizationParams);
            }
          }

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_LE(u32(c[mIndex * cStride() + nIndex]), u32(qmax()));
              ASSERT_GE(u32(c[mIndex * cStride() + nIndex]), u32(qmin()));
              ASSERT_EQ(c[mIndex * cStride() + nIndex], cRef[mIndex * n() + nIndex])
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << (u32)cRef[mIndex * n() + nIndex]
                  << ", optimized = " << (u32)c[mIndex * cStride() + nIndex]
                  << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
        }
        */
    }
    
    pub fn test(&self, hgemm: PyTorchHGemmUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());
        ASSERT_GE(k(), kr());
        ASSERT_GE(aStride(), k());
        ASSERT_GE(cStride(), n());

        random_device randomDevice;
        auto rng = bind(
            fp16_ieee_from_fp32_value,
            bind(
                uniform_real_distribution<float>(),
                mt19937(randomDevice())));

        vector<u16> a((m() - 1) * aStride() + k() + 4);
        vector<u16> b(n() * k());
        vector<u16, AlignedAllocator<u16, 32>> packedW(
            packedN() * packedK() + biasN());
        vector<u16> bias(n());
        vector<u16> c((mr() - 1) * cStride() + nr());
        vector<float> cRef(m() * n());

        const u16* aPtr = a.data() + 4;

        struct pytorch_qnnp_fp16_clamping_params clampingParams;
        clampingParams.scale = UINT16_C(0x3C00) /* 1.0 */;

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(rng));
          generate(b.begin(), b.end(), ref(rng));
          generate(bias.begin(), bias.end(), ref(rng));
          fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);
          fill(cRef.begin(), cRef.end(), 0.0f);

          fill(packedW.begin(), packedW.end(), 0);
          pytorch_pack_hgemm_w(n(), k(), np(), kr(), b.data(), bias.data(), packedW.data());

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize kBlockStart = 0; kBlockStart < k(); kBlockStart += kr()) {
                for (usize kBlockOffset = 0;
                     kBlockOffset < min(k() - kBlockStart, kr());
                     kBlockOffset++) {
                  ASSERT_LE(n(), packedN());
                  ASSERT_LT(mIndex * n() + nIndex, cRef.size());
                  ASSERT_LT(mIndex * k() + kBlockStart + kBlockOffset, a.size());

                  cRef[mIndex * n() + nIndex] +=
                      fp16_ieee_to_fp32_value(
                          aPtr[mIndex * aStride() + kBlockStart + kBlockOffset]) *
                      fp16_ieee_to_fp32_value(
                          b[nIndex * k() + kBlockStart + kBlockOffset]);
                }
              }
              cRef[mIndex * n() + nIndex] += fp16_ieee_to_fp32_value(bias[nIndex]);
            }
          }

          const float accMin = *min_element(cRef.cbegin(), cRef.cend());
          const float accMax = *max_element(cRef.cbegin(), cRef.cend());
          const float cMin = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
              accMin + (accMax - accMin) / 255.0f * float(qmin())));
          const float cMax = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
              accMax - (accMax - accMin) / 255.0f * float(255 - qmax())));
          clampingParams.max = fp16_ieee_from_fp32_value(cMax);
          clampingParams.min = fp16_ieee_from_fp32_value(cMin);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              cRef[mIndex * n() + nIndex] =
                  max(min(cRef[mIndex * n() + nIndex], cMax), cMin);
            }
          }

          hgemm(
              m(),
              n(),
              k(),
              aPtr,
              aStride() * sizeof(u16),
              packedW.data(),
              c.data(),
              cStride() * sizeof(u16),
              &clampingParams);

          /* Validate micro-kernel outputs */
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_NEAR(
                  fp16_ieee_to_fp32_value(c[mIndex * cStride() + nIndex]),
                  cRef[mIndex * n() + nIndex],
                  abs(cRef[mIndex * n() + nIndex]) * 1.0e-2f)
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << cRef[mIndex * n() + nIndex]
                  << ", optimized = "
                  << fp16_ieee_to_fp32_value(c[mIndex * cStride() + nIndex])
                  << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
          /* Check that micro-kernel did not overwrite data beyond bounds */
          for (usize mIndex = 0; mIndex < m() - 1; mIndex++) {
            for (usize nIndex = n(); nIndex < cStride(); nIndex++) {
              ASSERT_EQ(UINT16_C(0x7E00) /* NaN */, c[mIndex * cStride() + nIndex])
                  << "at " << mIndex << ", " << nIndex
                  << ": Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
          for (usize i = (m() - 1) * cStride() + n(); i < c.size(); i++) {
            ASSERT_EQ(UINT16_C(0x7E00) /* NaN */, c[i])
                << "at i = " << i << ", Mr x Nr x Kr = " << mr() << " x " << nr()
                << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x "
                << k();
          }
        }
        */
    }
    
    pub fn test(&self, sgemm: PyTorchSGemmUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());
        ASSERT_GE(k(), kr());
        ASSERT_GE(aStride(), k());
        ASSERT_GE(cStride(), n());

        random_device randomDevice;
        auto rng = bind(
            uniform_real_distribution<float>(), mt19937(randomDevice()));

        vector<float> a((m() - 1) * aStride() + k());
        vector<float> b(n() * k());
        vector<float> bias(n());
        vector<float, AlignedAllocator<float, 32>> packedW(
            packedN() * packedK() + biasN());
        vector<float> c((mr() - 1) * cStride() + nr());
        vector<float> cRef(m() * n());

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(rng));
          generate(b.begin(), b.end(), ref(rng));
          generate(bias.begin(), bias.end(), ref(rng));
          fill(c.begin(), c.end(), nanf(""));
          fill(cRef.begin(), cRef.end(), 0.0f);

          fill(packedW.begin(), packedW.end(), 0.0f);
          pytorch_pack_sgemm_w(n(), k(), np(), kr(), b.data(), bias.data(), packedW.data());

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize kIndex = 0; kIndex < k(); kIndex++) {
                ASSERT_LE(n(), packedN());
                ASSERT_LT(mIndex * n() + nIndex, cRef.size());
                cRef[mIndex * n() + nIndex] +=
                    a[mIndex * aStride() + kIndex] * b[nIndex * k() + kIndex];
              }
              cRef[mIndex * n() + nIndex] += bias[nIndex];
            }
          }

          const float accMin = *min_element(cRef.cbegin(), cRef.cend());
          const float accMax = *max_element(cRef.cbegin(), cRef.cend());
          const float cMin = accMin + (accMax - accMin) / 255.0f * float(qmin());
          const float cMax =
              accMax - (accMax - accMin) / 255.0f * float(255 - qmax());
          struct pytorch_qnnp_fp32_clamping_params clampingParams = {
              .max = cMax,
              .min = cMin,
          };

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              cRef[mIndex * n() + nIndex] =
                  max(min(cRef[mIndex * n() + nIndex], cMax), cMin);
            }
          }

          sgemm(
              m(),
              n(),
              k(),
              a.data(),
              aStride() * sizeof(float),
              packedW.data(),
              c.data(),
              cStride() * sizeof(float),
              &clampingParams);

          /* Validate micro-kernel outputs */
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_NEAR(
                  c[mIndex * cStride() + nIndex],
                  cRef[mIndex * n() + nIndex],
                  abs(cRef[mIndex * n() + nIndex]) * 1.0e-6f)
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << cRef[mIndex * n() + nIndex]
                  << ", optimized = " << c[mIndex * cStride() + nIndex]
                  << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
          /* Check that micro-kernel did not overwrite data beyond bounds */
          for (usize mIndex = 0; mIndex < m() - 1; mIndex++) {
            for (usize nIndex = n(); nIndex < cStride(); nIndex++) {
              ASSERT_TRUE(isnan(c[mIndex * cStride() + nIndex]))
                  << "at " << mIndex << ", " << nIndex
                  << ": Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x K = " << m() << " x " << n() << " x " << k();
            }
          }
          for (usize i = (m() - 1) * cStride() + n(); i < c.size(); i++) {
            ASSERT_TRUE(isnan(c[i]))
                << "at i = " << i << ", Mr x Nr x Kr = " << mr() << " x " << nr()
                << " x " << kr() << ", M x N x K = " << m() << " x " << n() << " x "
                << k();
          }
        }
        */
    }
    
    pub fn test(&self, sconv: PyTorchSConvUKernelFunction)  {
        
        todo!();
        /*
            ASSERT_LE(m(), mr());
        ASSERT_LE(n(), nr());
        ASSERT_GE(k(), kr());

        random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto f32rng = bind(
            uniform_real_distribution<float>(), mt19937(randomDevice()));

        vector<float> a((mr() - 1) * aStride() + k() + 8);
        vector<float> b(n() * ks() * k());
        vector<float, AlignedAllocator<float, 32>> packedW(
            ks() * packedK() * packedN() + biasN());
        vector<float> bias(n());
        vector<float> c((m() - 1) * cStride() + n());
        vector<float> cRef(m() * n());
        vector<const float*> im2col(mr() * ks());

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(a.begin(), a.end(), ref(f32rng));
          generate(b.begin(), b.end(), ref(f32rng));
          generate(bias.begin(), bias.end(), ref(f32rng));
          fill(c.begin(), c.end(), nanf(""));
          fill(cRef.begin(), cRef.end(), 0.0f);

          fill(packedW.begin(), packedW.end(), 0.0f);
          pytorch_pack_sconv_w(
              n(), ks(), k(), np(), kr(), b.data(), bias.data(), packedW.data());

          ASSERT_NE(
              *max_element(a.cbegin(), a.cend()),
              *min_element(a.cbegin(), a.cend()));
          ASSERT_NE(
              *max_element(b.cbegin(), b.cend()),
              *min_element(b.cbegin(), b.cend()));

          for (usize ksIndex = 0; ksIndex < ks(); ksIndex++) {
            for (usize mIndex = 0; mIndex < mr(); mIndex++) {
              im2col[ksIndex * mr() + mIndex] = a.data() + aStride() * mIndex;
            }
          }
          shuffle(im2col.begin(), im2col.end(), rng);
          for (usize ksIndex = 0; ksIndex < ks(); ksIndex++) {
            for (usize mIndex = m(); mIndex < mr(); mIndex++) {
              im2col[ksIndex * mr() + mIndex] = im2col[ksIndex * mr() + m() - 1];
            }
          }

          fill(cRef.begin(), cRef.end(), 0.0);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              for (usize ksIndex = 0; ksIndex < ks(); ksIndex++) {
                for (usize kBlockStart = 0; kBlockStart < k();
                     kBlockStart += kr()) {
                  for (usize kBlockOffset = 0;
                       kBlockOffset < min(k() - kBlockStart, kr());
                       kBlockOffset++) {
                    ASSERT_LT(ksIndex * mr() + mIndex, im2col.size());
                    ASSERT_LT(kBlockStart + kBlockOffset, k());
                    ASSERT_LT(kBlockStart + kBlockOffset, aStride());

                    cRef[mIndex * n() + nIndex] +=
                        double(im2col[ksIndex * mr() + mIndex]
                                     [kBlockStart + kBlockOffset]) *
                        double(
                            b[(nIndex * ks() + ksIndex) * k() + kBlockStart +
                              kBlockOffset]);
                  }
                }
              }
              cRef[mIndex * n() + nIndex] += bias[nIndex];
            }
          }

          const float accMin = *min_element(cRef.cbegin(), cRef.cend());
          const float accMax = *max_element(cRef.cbegin(), cRef.cend());
          if (m() * n() >= 3) {
            ASSERT_NE(accMax, accMin)
                << "Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                << ", M x N x K = " << m() << " x " << n() << " x " << k();
          }

          const float cRefMin = accMin + float(qmin()) / 255.0f * (accMax - accMin);
          const float cRefMax =
              accMax - float(255 - qmax()) / 255.0f * (accMax - accMin);
          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              cRef[mIndex * n() + nIndex] =
                  min(cRef[mIndex * n() + nIndex], cRefMax);
              cRef[mIndex * n() + nIndex] =
                  max(cRef[mIndex * n() + nIndex], cRefMin);
            }
          }

          const struct pytorch_qnnp_fp32_clamping_params clampingParams {
            cRefMax, cRefMin
          };

          sconv(
              m(),
              n(),
              k(),
              ks(),
              im2col.data(),
              packedW.data(),
              c.data(),
              cStride() * sizeof(float),
              &clampingParams);

          for (usize mIndex = 0; mIndex < m(); mIndex++) {
            for (usize nIndex = 0; nIndex < n(); nIndex++) {
              ASSERT_LE(c[mIndex * cStride() + nIndex], cRefMax);
              ASSERT_GE(c[mIndex * cStride() + nIndex], cRefMin);
              ASSERT_NEAR(
                  c[mIndex * cStride() + nIndex],
                  cRef[mIndex * n() + nIndex],
                  abs(cRef[mIndex * n() + nIndex]) * 1.0e-6f)
                  << "at " << mIndex << ", " << nIndex
                  << ": reference = " << cRef[mIndex * n() + nIndex]
                  << ", optimized = " << c[mIndex * cStride() + nIndex]
                  << ", Mr x Nr x Kr = " << mr() << " x " << nr() << " x " << kr()
                  << ", M x N x KC x KS = " << m() << " x " << n() << " x " << k()
                  << " x " << ks();
            }
          }
        }
        */
    }
}
