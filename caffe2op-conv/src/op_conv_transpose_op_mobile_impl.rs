crate::ix!();

#[inline] pub fn run_tile_contiguous<T, Context>(
    tile_id:           i32,
    n:                 i32,
    m:                 i32,
    h:                 i32,
    w:                 i32,
    output_h:          i32,
    output_w:          i32,
    c:                 i32,
    kernel_h:          i32,
    kernel_w:          i32,
    stride_h:          i32,
    stride_w:          i32,
    padT:              i32,
    filter_data:       *const T,
    xdata:             *const T,
    col_buffer_data:   *mut T,
    ydata:             *mut T,
    context:           *mut Context) 
{
    /**
      | The tile size is exactly the length of
      | a single row
      |
      */
      let tile_size: i32 = w;

      let kernel_data_size = c * kernel_h * kernel_w;
      let current_tile_start = tile_size * tile_id;

    todo!();
    /*
      // gemm tile
      math::GemmEx<T, Context>(
          CblasTrans,
          CblasNoTrans,
          kernel_data_size,
          tile_size,
          M,
          1,
          filterData,
          kernel_data_size,
          Xdata + current_tile_start,
          H * W,
          0,
          colBufferData,
          tile_size,
          context);

      // col2im tile
      // We assume that there is no padding in the columns (padL and padR
      // == 0).
      // FIXME: it is actually possible for us to handle padding, figure
      // out how to adjust the bounds

      // We write into Y in a de-interleaved fashion; in other words,
      // every column (mod stride_w) == 0 together in one block,
      // every column (mod stride_w) == 1 in another,
      // ... and so on.
      int colBlockSize = (W + kernel_w / stride_w);
      int numColBlocks = stride_w;

      for (int c = 0; c < kernel_data_size; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;

        // Each row is a separate tile that we handle. First determine the
        // row into which we are writing the output.
        // We can properly handle padding for the rows.
        int rowY = tile_id * stride_h - padT + h_offset;

        // If this row is out of bounds, then skip it
        if (!math::utils::IsAGeZeroAndALtB(rowY, output_h)) {
          continue;
        }

        // FIXME: we don't actually handle a dynamic padL > 0
        constexpr int kPadL = 0;
        int colOffsetStart = -kPadL + w_offset;
        int colBlockY = colOffsetStart % stride_w;

        // However, within a block we may not start writing at offset
        // 0. The offset at which we begin writing is determined by
        // colOffsetStart
        int colWithinBlockOffsetY = colOffsetStart / stride_w;

        // So, this is where we begin reading/writing in Y
        int colY = colBlockY * colBlockSize + colWithinBlockOffsetY;

        // This is the complete offset into Y from the start
        // Each row has stride_w blocks of size colBlockSize
        int offsetY = rowY * colBlockSize * numColBlocks + colY;

        T* colBufferPointer = colBufferData + c * tile_size;
        T* yPointer =
            Ydata + c_im * output_h * (colBlockSize * numColBlocks) + offsetY;

        int b = 0;
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        // We vectorize the loop within the row
        {
          constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float)) * 4;
          int limit = (tile_size / kUnroll) * kUnroll;

          for (; b < limit; b += kUnroll) {
            float32x4_t cb0 = vld1q_f32(colBufferPointer + 0);
            float32x4_t cb1 = vld1q_f32(colBufferPointer + 4);
            float32x4_t cb2 = vld1q_f32(colBufferPointer + 8);
            float32x4_t cb3 = vld1q_f32(colBufferPointer + 12);

            float32x4_t y0 = vld1q_f32(yPointer + 0);
            float32x4_t y1 = vld1q_f32(yPointer + 4);
            float32x4_t y2 = vld1q_f32(yPointer + 8);
            float32x4_t y3 = vld1q_f32(yPointer + 12);

            y0 = vaddq_f32(y0, cb0);
            y1 = vaddq_f32(y1, cb1);
            y2 = vaddq_f32(y2, cb2);
            y3 = vaddq_f32(y3, cb3);

            vst1q_f32(yPointer + 0, y0);
            vst1q_f32(yPointer + 4, y1);
            vst1q_f32(yPointer + 8, y2);
            vst1q_f32(yPointer + 12, y3);

            colBufferPointer += kUnroll;
            yPointer += kUnroll;
          }
        }

        {
          constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float));
          int limit = (tile_size / kUnroll) * kUnroll;

          for (; b < limit; b += kUnroll) {
            float32x4_t cb0 = vld1q_f32(colBufferPointer);
            float32x4_t y0 = vld1q_f32(yPointer);

            y0 = vaddq_f32(y0, cb0);

            vst1q_f32(yPointer, y0);

            colBufferPointer += kUnroll;
            yPointer += kUnroll;
          }
        }
    #endif

        // Handle un-vectorizable epilogue
        for (; b < tile_size; ++b) {
          *yPointer += *colBufferPointer;
          ++yPointer;
          ++colBufferPointer;
        }
      }
    */
}

#[inline] pub fn reinterleave_rows<const kStrideW: i32>(
    src:         *const f32,
    bias:        *const f32,
    c:           i32,
    h:           i32,
    dst:         *mut f32,
    outputC:     i32,
    output_h:    i32,
    output_w:    i32,
    inputW:      i32,
    kernel_w:    i32,
    stride_w:    i32,
    adjH:        i32) 
{
    todo!();
    /*
        // Each row in src is of the form:
      // [w mod stride_w == 0 elements]...[w mod stride_w == stride_w - 1
      // elements]
      // We need to re-interleave the values and write them in the output
      int colBlockSize = inputW + kernel_w / kStrideW;
      int noAdjOutputW = (inputW - 1) * kStrideW + kernel_w;

      int point = c * output_h + h;
      src += point * colBlockSize * kStrideW;
      dst += point * output_w;

      float b = bias ? bias[c] : 0;
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      float32x4_t biasV = vdupq_n_f32(b);
    #endif

      int w = 0;
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float)) * 2;
      int limit = ((inputW - 1) / kUnroll) * kUnroll;

      for (; w < limit; w += kUnroll) {
        // We need to interleave in terms of kStrideW units
        float32x4_t v0[kStrideW];
        float32x4_t v1[kStrideW];

        for (int i = 0; i < kStrideW; ++i) {
          v0[i] = vld1q_f32(src + i * colBlockSize);
          v1[i] = vld1q_f32(src + i * colBlockSize + 4);
        }

        // add per-channel bias
        for (int i = 0; i < kStrideW; ++i) {
          v0[i] = vaddq_f32(v0[i], biasV);
          v1[i] = vaddq_f32(v1[i], biasV);
        }

        // Write interleaved into the output
        StoreInterleaved<float, kStrideW>::store(dst + 0 * kStrideW, v0);
        StoreInterleaved<float, kStrideW>::store(dst + 4 * kStrideW, v1);

        src += kUnroll;
        dst += kUnroll * kStrideW;
      }
    #endif

      // Handle non-vectorizable remainder
      for (; w < inputW - 1; ++w) {
        float v[kStrideW];

        for (int i = 0; i < kStrideW; ++i) {
          v[i] = src[i * colBlockSize];
        }

        // add per-channel bias
        for (int i = 0; i < kStrideW; ++i) {
          v[i] += b;
        }

        // Write interleaved into the output
        StoreInterleaved<float, kStrideW>::store(dst, v);

        src += 1;
        dst += kStrideW;
      }

      // We have handled 0 .. (inputW - 1) * stride inclusive so far.
      // Handle the remainder
      int outputPoint = (inputW - 1) * kStrideW;
      int block = 0;

      // Output width may include adjustment into which we don't
      // write; ignore it
      while (outputPoint < noAdjOutputW) {
        float v = src[block * colBlockSize];
        dst[0] = v + b;
        ++outputPoint;
        dst += 1;

        ++block;
        if (block >= kStrideW) {
          block = 0;
          src += 1;
        }
      }

      // Remainder of the buffer comprised of just the `adj` must have
      // bias added
      for (; outputPoint < output_w; ++outputPoint) {
        dst[0] = b;
        dst += 1;
      }
    */
}

#[inline] pub fn reinterleave_multithreaded<const N: i32, T, Context>(
    y0:         *const T,
    bias_data:  *const T,
    y:          *mut T,
    outputC:    i32,
    output_h:    i32,
    output_w:    i32,
    inputW:     i32,
    kernel_w:    i32,
    stride_w:    i32,
    adjH:       i32,
    pool:       *mut ThreadPool) 
{
    todo!();
    /*
        // # channels times height
      size_t totalTiles = (size_t)outputC * output_h;
      FixedDivisor<int> divOutputH(output_h);

    #define REINTERLEAVE(N)  \
      do {                   \
        reinterleaveRows<N>( \
            y0,              \
            bias_data,       \
            c,               \
            h,               \
            y,               \
            outputC,         \
            output_h,         \
            output_w,         \
            inputW,          \
            kernel_w,         \
            stride_w,         \
            adjH);           \
      } while (false)

      std::function<void(int, size_t)> fnReinterleave = [&](int threadId,
                                                            size_t tile_id) {
        int h;
        int c;
        divOutputH.DivMod((int)tile_id, &c, &h);

        REINTERLEAVE(N);
      };

    #undef REINTERLEAVE

      pool->run(fnReinterleave, totalTiles);
    */
}

impl<T, Context> ConvTransposeMobileOp<T, Context> {

    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const Tensor& X = Input(INPUT);
        auto& filter = Input(FILTER);
        const int N = X.dim32(0), M = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
        CAFFE_ENFORCE(filter.ndim() == 4, "filter must be 4D tensor");
        CAFFE_ENFORCE(
            filter.dim32(0) == M,
            "filter number must be equal to input channel number");
        const int C = filter.dim32(1);
        CAFFE_ENFORCE(
            filter.dim32(2) == this->kernel_h(),
            "filter height must be equal to kernel height");
        CAFFE_ENFORCE(
            filter.dim32(3) == this->kernel_w(),
            "filter width must be equal to kernel width");
        if (InputSize() == 3) {
            auto& bias = Input(BIAS);
            CAFFE_ENFORCE(bias.ndim() == 1, "bias must be 1D tensor");
            CAFFE_ENFORCE(
                bias.dim32(0) == C,
                "bias dimension must be equal to output channel number");
        }

        auto sizes = ConvTransposeUnpoolBase<Context>::GetOutputSize(X, C);
        Tensor* Y = Output(0, sizes, at::dtype<T>());

        if (X.numel() == 0) {
            VLOG(2) << "Number of elements is 0 in ConvTrasposeOp";
            return true;
        }

        const int output_h = Y->dim32(2);
        const int output_w = Y->dim32(3);
        const int outputPlaneSize = output_h * output_w;
        const int outputBatchElementSize = Y->dim32(1) * outputPlaneSize;

        auto Xdata = X.template data<T>();
        auto Ydata = Y->template mutable_data<T>();

        auto pool = ws_->GetThreadPool();
        auto numThreads = pool->getNumThreads();

        // Initialize per-thread buffers for output
        // The main thread will write directly into the output Y, we just
        // need buffers for the worker threads
        size_t colBlockSize = W + this->kernel_w() / this->stride_w();
        size_t threadYBufferSize = C * output_h * colBlockSize * this->stride_w();
        // Require 16 byte alignment, so 4-element alignment as these are floats.
        size_t threadYBufferSizeAligned =
            ((C * output_h * colBlockSize * this->stride_w() + 3) / 4) * 4;
        size_t threadColBufferSize = C * this->kernel_h() * this->kernel_w() * W;

        // Work around GCC 4.9 bug when this is declared inside the inner lambda.
        auto runLocalTile = [&](TensorCPU* threadBuffer,
            int threadId,
            size_t tile_id) {
            auto localYData = threadBuffer->template mutable_data<T>() +
                threadId * threadYBufferSizeAligned;

            auto localColBufferData = threadBuffer->template mutable_data<T>() +
                numThreads * threadYBufferSizeAligned + threadId * threadColBufferSize;

            runTileContiguous<T, Context>(
                tile_id,
                N,
                M,
                H,
                W,
                output_h,
                output_w,
                C,
                this->kernel_h(),
                this->kernel_w(),
                this->stride_h(),
                this->stride_w(),
                this->pad_t(),
                filter.template data<T>(),
                Xdata,
                localColBufferData,
                localYData,
                &context_);
        };

        auto f = [&](Tensor* threadBuffer) {
            threadBuffer->Resize(
                numThreads * threadYBufferSizeAligned +
                numThreads * threadColBufferSize);
            // Group together thread buffers for accumulation
            std::vector<T*> toSum(numThreads - 1);
            for (int i = 1; i < numThreads; ++i) {
                toSum[i - 1] = threadBuffer->template mutable_data<T>() +
                    i * threadYBufferSizeAligned;
            }

            for (auto image_id = 0; image_id < N; ++image_id) {
                // Each time through, we have to reset all per-thread output
                // buffers, since the output buffer is only per-batch element
                // The column buffers are overwritten by the matrix multiplication
                // each time, so we need not clear them out each round
                math::Set<T, Context>(
                    numThreads * threadYBufferSizeAligned,
                    0,
                    threadBuffer->template mutable_data<T>(),
                    &context_);

                // Run tiled gemm and col2im in our threadpool; all of these tiles
                // are guaranteed to be full tiles
                // Each tile handles a single row of the input
                pool->run(
                    [&](int threadId, int tile_id) {
                        runLocalTile(threadBuffer, threadId, tile_id);
                    },
                    H);

                // We need to accumulate the per-thread results into the output
                // Y; the first worker thread (main thread) already produced its
                // results in Y
                sumInto(
                    threadBuffer->template mutable_data<T>(), toSum, threadYBufferSize);

                /// y0 now contains the final output, but it is in deinterleaved
                /// form. We have to re-interleave it to produce the final form in Y
                /// This operation also handles adding the per-channel bias.
                macro_rules! reinterleave {
                    ($N:ident) => {
                        reinterleaveMultithreaded<N, T, Context>(                        
                            threadBuffer->template mutable_data<T>(),                    
                            InputSize() == 3 ? Input(BIAS).template data<T>() : nullptr, 
                            Ydata,                                                       
                            Y->dim32(1),                                                 
                            Y->dim32(2),                                                 
                            Y->dim32(3),                                                 
                            W,                                                           
                            this->kernel_w(),                                            
                            this->stride_w(),                                            
                            this->adj_h(),                                               
                            pool);                                                       
                    }
                }

                if (this->stride_w() == 1) {
                    REINTERLEAVE(1);
                } else if (this->stride_w() == 2) {
                    REINTERLEAVE(2);
                } else if (this->stride_w() == 3) {
                    REINTERLEAVE(3);
                } else if (this->stride_w() == 4) {
                    REINTERLEAVE(4);
                }

                Xdata += M * H * W;
                Ydata += Y->size() / Y->dim32(0);
            }
        };
        if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
            runWithSharedBuffer<Context>(ws_, f);
        } else {
            f(&threadBuffer_);
        }

        return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_THROW("Not implemented.");
        */
    }
}
