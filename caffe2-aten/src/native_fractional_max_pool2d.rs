// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/FractionalMaxPool2d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(fractional_max_pool2d) (
      const Tensor& input,
      IntArrayRef pool_size,
      IntArrayRef output_size,
      const Tensor& randomSamples
    ) {
      TORCH_CHECK(
          pool_size.size() == 2,
          "fractional_max_pool2d: kernel_size must either be a single Int or tuple of Ints")
      TORCH_CHECK(
          output_size.size() == 2,
          "fractional_max_pool2d: output_size must either be a single Int or tuple of Ints")
      i64 numBatch = 1;
      i64 planeDim = 0;
      i64 heightDim = 1;
      i64 widthDim = 2;
      i64 outputH = output_size[0];
      i64 outputW = output_size[1];
      i64 poolSizeH = pool_size[0];
      i64 poolSizeW = pool_size[1];

      i64 ndims = input.ndimension();
      TORCH_CHECK(input.numel() > 0 && (ndims == 3 || ndims == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input, but got: ",
        ndims);

      if (ndims == 4) {
        numBatch = input.size(0);
        planeDim++;
        heightDim++;
        widthDim++;
      }

      /* sizes */
      i64 numPlanes = input.size(planeDim);
      i64 inputH = input.size(heightDim);
      int inputW = input.size(widthDim);

      TORCH_CHECK(outputH + poolSizeH - 1 <= inputH,
        "fractional_max_pool2d(): pool height ", poolSizeH,
        " too large relative to input height ", inputH);
      TORCH_CHECK(outputW + poolSizeW - 1 <= inputW,
        "fractional_max_pool2d(): pool width ", poolSizeW,
        " too large relative to input width ", inputW);

      if (ndims == 3) {
        set_output(0, {numPlanes, outputH, outputW}, input.options());
        /* indices will contain the locations for each output point */
        set_output(1, {numPlanes, outputH, outputW}, input.options().dtype(kLong));
      } else {
        set_output(0, {numBatch, numPlanes, outputH, outputW}, input.options());
        /* indices will contain the locations for each output point */
        set_output(1, {numBatch, numPlanes, outputH, outputW}, input.options().dtype(kLong));
      }
    }
    */
}


pub fn fractional_max_pool2d_generate_intervals<Scalar>(
        sample:      Scalar,
        input_size:  i32,
        output_size: i32,
        pool_size:   i32) -> Vec<i32> {

    todo!();
        /*
            vector<int> sequence(outputSize);
      if (outputSize > 1) {
        Scalar alpha = static_cast<Scalar>(inputSize - poolSize) /
          static_cast<Scalar>(outputSize - 1);

        for (int i = 0; i < outputSize - 1; ++i) {
          sequence[i] =
            static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha);
        }
      }
      sequence[outputSize - 1] = inputSize - poolSize;

      return sequence;
        */
}

pub fn fractional_max_pool2d_out_single_batch_frame<Scalar>(
        input:          *mut Scalar,
        output:         *mut Scalar,
        indices:        *mut i64,
        random_samples: *mut Scalar,
        num_planes:     i32,
        inputw:         i32,
        inputh:         i32,
        outputw:        i32,
        outputh:        i32,
        pool_sizew:     i32,
        pool_sizeh:     i32)  {

    todo!();
        /*
            parallel_for(0, numPlanes, 0, [&](i64 start, i64 end) {
        for (auto plane = start; plane < end; ++plane) {
          /* each plane contains 2 random samples, one for W and one for H */
          Scalar* randomSamplesForPlane = randomSamples + plane * 2;

          /* Generate interval sequence */
          auto sequenceW = fractional_max_pool2d_generate_intervals<Scalar>(
              randomSamplesForPlane[0], inputW, outputW, poolSizeW);
          auto sequenceH = fractional_max_pool2d_generate_intervals<Scalar>(
              randomSamplesForPlane[1], inputH, outputH, poolSizeH);

          /* loop over output */
          int h, w;

          Scalar* inputForPlane = input + plane * inputW * inputH;
          Scalar* outputForPlane = output + plane * outputW * outputH;
          i64* indicesForPlane = indices + plane * outputW * outputH;

          for (h = 0; h < outputH; ++h) {
            int inputHStart = sequenceH[h];

            for (w = 0; w < outputW; ++w) {
              int inputWStart = sequenceW[w];

              int h2 = inputHStart, w2 = inputWStart;
              Scalar maxVal = -numeric_limits<Scalar>::infinity();
              i64 maxIndex = h2 * inputW + w2;

              for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
                for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
                  AT_ASSERT(h2 >= 0 && h2 < inputH);
                  AT_ASSERT(w2 >= 0 && w2 < inputW);

                  int planeIndex = h2 * inputW + w2;
                  Scalar val = inputForPlane[planeIndex];
                  if (val > maxVal || isnan(val)) {
                    maxVal = val;
                    maxIndex = planeIndex;
                  }
                }
              }

              outputForPlane[h * outputW + w] = maxVal;
              indicesForPlane[h * outputW + w] = maxIndex;
            }
          }
        }
      });
        */
}

pub fn fractional_max_pool2d_out_frame<Scalar>(
        input:          *mut Scalar,
        output:         *mut Scalar,
        indices:        *mut i64,
        random_samples: *mut Scalar,
        num_batch:      i32,
        num_planes:     i32,
        inputw:         i32,
        inputh:         i32,
        outputw:        i32,
        outputh:        i32,
        pool_sizew:     i32,
        pool_sizeh:     i32)  {

    todo!();
        /*
            if(numBatch == 1) {
          fractional_max_pool2d_out_single_batch_frame<Scalar>(
            input,
            output,
            indices,
            randomSamples,
            numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH
          );
          return;
        }
        parallel_for(0, numBatch, 0, [&](i64 start, i64 end) {
          for (auto batch = start; batch < end; ++batch) {
            fractional_max_pool2d_out_single_batch_frame<Scalar>(
              input + batch * numPlanes * inputH * inputW,
              output + batch * numPlanes * outputH * outputW,
              indices + batch * numPlanes * outputH * outputW,
              randomSamples + batch * numPlanes * 2,
              numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
          }
        });
        */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(fractional_max_pool2d_out_cpu) (
      const Tensor& input_,
      IntArrayRef pool_size,
      IntArrayRef output_size,
      const Tensor& randomSamples,
      const Tensor& output,
      const Tensor& indices) {

      i64 numBatch = 1;
      i64 planeDim = 0;
      i64 heightDim = 1;
      i64 widthDim = 2;
      i64 outputH = output_size[0]; // output.size(heightDim)
      i64 outputW = output_size[1]; // output.size(widthDim)
      i64 poolSizeH = pool_size[0];
      i64 poolSizeW = pool_size[1];

      /* get contiguous input */
      auto input = input_.contiguous();

      i64 ndims = input.ndimension();

      if (ndims == 4) {
        numBatch = input.size(0);
        planeDim++;
        heightDim++;
        widthDim++;
      }

      /* sizes */
      i64 numPlanes = input.size(planeDim);
      i64 inputH = input.size(heightDim);
      i64 inputW = input.size(widthDim);

      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "fractional_max_pool2d_out_frame", [&] {
        auto input_data = input.data_ptr<Scalar>();
        auto output_data = output.data_ptr<Scalar>();
        auto indices_data = indices.data_ptr<i64>();
        auto randomSamples_data = randomSamples.data_ptr<Scalar>();
        fractional_max_pool2d_out_frame<Scalar>(
          input_data,
          output_data,
          indices_data,
          randomSamples_data,
          numBatch, numPlanes,
          inputW, inputH,
          outputW, outputH,
          poolSizeW, poolSizeH);
        }
      );
    }
    */
}


pub fn fractional_max_pool2d_backward_out_single_batch_frame<Scalar>(
        grad_input:  *mut Scalar,
        grad_output: *mut Scalar,
        indices:     *mut i64,
        num_planes:  i32,
        inputw:      i32,
        inputh:      i32,
        outputw:     i32,
        outputh:     i32)  {

    todo!();
        /*
            parallel_for(0, numPlanes, 0, [&](i64 start, i64 end) {
        for (auto plane = start; plane < end; plane++) {
          Scalar* gradInputForPlane = gradInput + plane * inputW * inputH;
          Scalar* gradOutputForPlane = gradOutput + plane * outputW * outputH;
          i64* indicesForPlane = indices + plane * outputW * outputH;

          int h, w;
          for (h = 0; h < outputH; ++h) {
            for (w = 0; w < outputW; ++w) {
              int outputIndex = h * outputW + w;
              i64 index = indicesForPlane[outputIndex];
              AT_ASSERT(index >= 0 && index < inputW * inputH);

              gradInputForPlane[index] += gradOutputForPlane[outputIndex];
            }
          }
        }
      });
        */
}

pub fn fractional_max_pool2d_backward_out_frame<Scalar>(
        grad_input:  *mut Scalar,
        grad_output: *mut Scalar,
        indices:     *mut i64,
        num_batch:   i32,
        num_planes:  i32,
        inputw:      i32,
        inputh:      i32,
        outputw:     i32,
        outputh:     i32)  {

    todo!();
        /*
            if(numBatch == 1) {
          fractional_max_pool2d_backward_out_single_batch_frame<Scalar>(
            gradInput, gradOutput, indices,
            numPlanes,
            inputW, inputH, outputW, outputH
          );
          return;
        }
        parallel_for(0, numBatch, 0, [&](i64 start, i64 end) {
          for (auto batch = start; batch < end; ++batch) {
            fractional_max_pool2d_backward_out_single_batch_frame<Scalar>(
              gradInput + batch * numPlanes * inputH * inputW,
              gradOutput + batch * numPlanes * outputH * outputW,
              indices + batch * numPlanes * outputH * outputW,
              numPlanes, inputW, inputH, outputW, outputH);
          }
        });
        */
}

pub fn fractional_max_pool2d_backward_out_cpu_template(
        input:       &Tensor,
        grad_output: &Tensor,
        grad_input:  &mut Tensor,
        output_size: &[i32],
        pool_size:   &[i32], /* unused */
        indices:     &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            int numBatch = 1;
      int planeDim = 0;
      int heightDim = 1;
      int widthDim = 2;

      int outputH = output_size[0];
      int outputW = output_size[1];

      int ndims = input.ndimension();
      if (ndims == 4) {
        numBatch = input.size(0);
        planeDim = 1;
        heightDim++;
        widthDim++;
      }

      /* sizes */
      int numPlanes = input.size(planeDim);
      int inputH = input.size(heightDim);
      int inputW = input.size(widthDim);

      /* get contiguous gradOutput */
      auto gradOutput = gradOutput_.contiguous();

      TORCH_CHECK(outputW == gradOutput.size(widthDim),
        "fractional_max_pool2d_backward(): gradOutput width unexpected");
      TORCH_CHECK(outputH == gradOutput.size(heightDim),
        "fractional_max_pool2d_backward(): gradOutput height unexpected");

      /* resize */
      gradInput.resize_as_(input);
      gradInput.zero_();

      /* backprop */
      AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "fractional_max_pool2d_backward_out_frame", [&] {
          auto gradInput_data = gradInput.data_ptr<Scalar>();
          auto gradOutput_data = gradOutput.data_ptr<Scalar>();
          auto indices_data = indices.data_ptr<i64>();
          fractional_max_pool2d_backward_out_frame<Scalar>(
            gradInput_data,
            gradOutput_data,
            indices_data,
            numBatch, numPlanes,
            inputW, inputH,
            outputW, outputH
          );
          }
        );
      return gradInput;
        */
}

pub fn fractional_max_pool2d_backward_out_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        pool_size:   &[i32],
        output_size: &[i32],
        indices:     &Tensor,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            gradInput.resize_as_(input);
      fractional_max_pool2d_backward_out_cpu_template(
        input,
        gradOutput_,
        gradInput,
        output_size,
        pool_size,
        indices);
      return gradInput;
        */
}

pub fn fractional_max_pool2d_backward_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        pool_size:   &[i32],
        output_size: &[i32],
        indices:     &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor gradInput = empty({0}, input.options());
      fractional_max_pool2d_backward_out_cpu_template(
        input,
        gradOutput_,
        gradInput,
        output_size,
        pool_size,
        indices);
      return gradInput;
        */
}
