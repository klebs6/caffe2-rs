crate::ix!();

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
