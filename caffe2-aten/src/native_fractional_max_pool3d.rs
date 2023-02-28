crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/FractionalMaxPool3d.cpp]

pub fn generate_intervals<Scalar>(
        sample:      Scalar,
        input_size:  i64,
        output_size: i64,
        pool_size:   i64) -> Vec<i32> {

    todo!();
        /*
            vector<int> sequence(outputSize);
      if (outputSize > 1) {
        Scalar alpha = static_cast<Scalar>(inputSize - poolSize) /
          static_cast<Scalar>(outputSize - 1);

        for (const auto i : irange(outputSize - 1)) {
          sequence[i] =
            static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha);
        }
      }
      sequence[outputSize - 1] = inputSize - poolSize;

      return sequence;
        */
}

pub fn fractional_max_pool3d_out_single_batch_frame<Scalar>(
        input:          *mut Scalar,
        output:         *mut Scalar,
        indices:        *mut i64,
        random_samples: *mut Scalar,
        num_planes:     i64,
        inputt:         i64,
        inputh:         i64,
        inputw:         i64,
        outputt:        i64,
        outputh:        i64,
        outputw:        i64,
        pool_sizet:     i64,
        pool_sizeh:     i64,
        pool_sizew:     i64)  {

    todo!();
        /*
            parallel_for(0, numPlanes, 0, [&](i64 start, i64 end) {
        for (auto plane = start; plane < end; ++plane) {
          /* each plane contains 3 random samples,
             one for T, one for W, and one for H */
          Scalar* randomSamplesForPlane = randomSamples + plane * 3;

          /* Generate interval sequence */
          auto sequenceT = generate_intervals<Scalar>(
              randomSamplesForPlane[0], inputT, outputT, poolSizeT);
          auto sequenceH = generate_intervals<Scalar>(
              randomSamplesForPlane[1], inputH, outputH, poolSizeH);
          auto sequenceW = generate_intervals<Scalar>(
              randomSamplesForPlane[2], inputW, outputW, poolSizeW);

          /* loop over output */
          i64 t, h, w;

          Scalar* inputForPlane = input + plane * inputT * inputH * inputW;
          Scalar* outputForPlane = output + plane * outputT * outputH * outputW;
          i64* indicesForPlane = indices + plane * outputT * outputH * outputW;

          for (t = 0; t < outputT; ++t) {
            i64 inputTStart = sequenceT[t];

            for (h = 0; h < outputH; ++h) {
              i64 inputHStart = sequenceH[h];

              for (w = 0; w < outputW; ++w) {
                i64 inputWStart = sequenceW[w];

                i64 t2 = inputTStart, h2 = inputHStart, w2 = inputWStart;
                Scalar maxVal = -numeric_limits<Scalar>::infinity();
                i64 maxIndex = t2 * inputH * inputW + h2 * inputW + w2;

                for (t2 = inputTStart; t2 < inputTStart + poolSizeT; ++t2) {
                  for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
                    for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
                      AT_ASSERT(t2 >= 0 && t2 < inputT);
                      AT_ASSERT(h2 >= 0 && h2 < inputH);
                      AT_ASSERT(w2 >= 0 && w2 < inputW);

                      i64 planeIndex = t2 * inputH * inputW + h2 * inputW + w2;
                      Scalar val = inputForPlane[planeIndex];
                      if (val > maxVal || isnan(val)) {
                        maxVal = val;
                        maxIndex = planeIndex;
                      }
                    }
                  }
                }

                outputForPlane[t * outputH * outputW + h * outputW + w] = maxVal;
                indicesForPlane[t * outputH * outputW + h * outputW + w] = maxIndex;
              }
            }
          }
        }
      });
        */
}

pub fn fractional_max_pool3d_out_frame<Scalar>(
        input:          *mut Scalar,
        output:         *mut Scalar,
        indices:        *mut i64,
        random_samples: *mut Scalar,
        num_batch:      i64,
        num_planes:     i64,
        inputt:         i64,
        inputh:         i64,
        inputw:         i64,
        outputt:        i64,
        outputh:        i64,
        outputw:        i64,
        pool_sizet:     i64,
        pool_sizeh:     i64,
        pool_sizew:     i64)  {

    todo!();
        /*
            if(numBatch == 1) {
          fractional_max_pool3d_out_single_batch_frame<Scalar>(
            input, output, indices, randomSamples,
            numPlanes,
            inputT, inputH, inputW,
            outputT, outputH, outputW,
            poolSizeT, poolSizeH, poolSizeW
          );
          return;
        }

        parallel_for(0, numBatch, 0, [&](i64 start, i64 end) {
          for (auto batch = start; batch < end; ++batch) {
            fractional_max_pool3d_out_single_batch_frame<Scalar>(
              input + batch * numPlanes * inputW * inputH * inputT,
              output + batch * numPlanes * outputW * outputH * outputT,
              indices + batch * numPlanes * outputW * outputH * outputT,
              randomSamples + batch * numPlanes * 3,
              numPlanes,
              inputT, inputH, inputW,
              outputT, outputH, outputW,
              poolSizeT, poolSizeH, poolSizeW
            );
          }
        });
        */
}

pub fn fractional_max_pool3d_out_cpu_template(
        output:         &mut Tensor,
        indices:        &mut Tensor,
        input:          &Tensor,
        pool_size:      &[i32],
        output_size:    &[i32],
        random_samples: &Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          pool_size.size() == 3,
          "fractional_max_pool3d: kernel_size must either be a single Int or tuple of three Ints")
      TORCH_CHECK(
          output_size.size() == 3,
          "fractional_max_pool3d: output_size must either be a single Int or tuple of three Ints")
      i64 outputT = output_size[0];
      i64 outputH = output_size[1];
      i64 outputW = output_size[2];
      i64 poolSizeT = pool_size[0];
      i64 poolSizeH = pool_size[1];
      i64 poolSizeW = pool_size[2];

      i64 numBatch = 1;
      i64 planeDim = 0;
      i64 timeDim = 1;
      i64 heightDim = 2;
      i64 widthDim = 3;

      i64 ndims = input_.ndimension();
      TORCH_CHECK(input_.numel() != 0 && (ndims == 4 || ndims == 5),
        "fractional_max_pool3d_out(): non-empty 4D or 5D (batch mode) tensor ",
        " expected for input, but got: ", ndims);

      if (ndims == 5) {
        numBatch = input_.size(0);
        planeDim++;
        timeDim++;
        heightDim++;
        widthDim++;
      }

      /* sizes */
      i64 numPlanes = input_.size(planeDim);
      i64 inputT = input_.size(timeDim);
      i64 inputH = input_.size(heightDim);
      i64 inputW = input_.size(widthDim);

      TORCH_CHECK(outputT + poolSizeT - 1 < inputT,
               "fractional_max_pool3d_out(): pool time ", poolSizeT,
               " too large relative to input time ", inputT);
      TORCH_CHECK(outputW + poolSizeW - 1 < inputW,
               "fractional_max_pool3d_out(): pool width ", poolSizeW,
               " too large relative to input width ", inputW);
      TORCH_CHECK(outputH + poolSizeH - 1 < inputH,
               "fractional_max_pool3d_out(): pool height ", poolSizeH,
               " too large relative to input height ", inputH);

      /* get contiguous input */
      auto input = input_.contiguous();

      if (ndims == 4) {
        /* resize output */
        output.resize_({numPlanes, outputT, outputH, outputW});
        /* indices will contain the locations for each output point */
        indices.resize_({numPlanes, outputT, outputH, outputW});
      } else {
        output.resize_({numBatch, numPlanes, outputT, outputH, outputW});
        /* indices will contain the locations for each output point */
        indices.resize_({numBatch, numPlanes, outputT, outputH, outputW});
      }
      AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "fractional_max_pool3d_out_frame",
        [&] {
          fractional_max_pool3d_out_frame<Scalar>(
            input.data_ptr<Scalar>(),
            output.data_ptr<Scalar>(),
            indices.data_ptr<i64>(),
            randomSamples.data_ptr<Scalar>(),
            numBatch, numPlanes,
            inputT, inputH, inputW,
            outputT, outputH, outputW,
            poolSizeT, poolSizeH, poolSizeW
          );
        }
      );
        */
}

pub fn fractional_max_pool3d_backward_out_single_batch_frame<Scalar>(
        grad_input:  *mut Scalar,
        grad_output: *mut Scalar,
        indices:     *mut i64,
        num_planes:  i64,
        inputt:      i64,
        inputh:      i64,
        inputw:      i64,
        outputt:     i64,
        outputh:     i64,
        outputw:     i64)  {

    todo!();
        /*
            parallel_for(0, numPlanes, 0, [&](i64 start, i64 end) {
        for (auto plane = start; plane < end; plane++) {
          Scalar* gradInputForPlane = gradInput + plane * inputT * inputH * inputW;
          Scalar* gradOutputForPlane = gradOutput +
                      plane * outputT * outputH * outputW;
          i64* indicesForPlane = indices + plane * outputT * outputH * outputW;

          i64 h, w, t;
          for (t = 0; t < outputT; ++t) {
            for (h = 0; h < outputH; ++h) {
              for (w = 0; w < outputW; ++w) {
                i64 outputIndex = t * outputH * outputW + h * outputW + w;
                i64 index = indicesForPlane[outputIndex];
                AT_ASSERT(index >= 0 && index < inputT * inputH * inputW);
                gradInputForPlane[index] += gradOutputForPlane[outputIndex];
              }
            }
          }
        }
      });
        */
}

pub fn fractional_max_pool3d_backward_out_frame<Scalar>(
        grad_input:  *mut Scalar,
        grad_output: *mut Scalar,
        indices:     *mut i64,
        num_batch:   i64,
        num_planes:  i64,
        inputt:      i64,
        inputh:      i64,
        inputw:      i64,
        outputt:     i64,
        outputh:     i64,
        outputw:     i64)  {

    todo!();
        /*
            if(numBatch == 1) {
          fractional_max_pool3d_backward_out_single_batch_frame<Scalar>(
            gradInput, gradOutput, indices,
            numPlanes,
            inputT, inputH, inputW,
            outputT, outputH, outputW
          );
          return;
        }

        parallel_for(0, numBatch, 0, [&](i64 start, i64 end) {
          for (auto batch = start; batch < end; ++batch) {
            fractional_max_pool3d_backward_out_single_batch_frame<Scalar>(
              gradInput + batch * numPlanes * inputW * inputH * inputT,
              gradOutput + batch * numPlanes * outputW * outputH * outputT,
              indices + batch * numPlanes * outputW * outputH * outputT,
              numPlanes,
              inputT, inputH, inputW,
              outputT, outputH, outputW
            );
          }
        });
        */
}

pub fn fractional_max_pool3d_backward_out_cpu_template(
        input:       &Tensor,
        grad_output: &Tensor,
        grad_input:  &mut Tensor,
        output_size: &[i32],
        pool_size:   &[i32], /* unused */
        indices:     &Tensor)  {
    
    todo!();
        /*
            i64 outputT = output_size[0];
      i64 outputH = output_size[1];
      i64 outputW = output_size[2];

      i64 numBatch = 1;
      i64 planeDim = 0;
      i64 timeDim = 1;
      i64 heightDim = 2;
      i64 widthDim = 3;

      i64 ndims = input.ndimension();
      if (ndims == 5) {
        numBatch = input.size(0);
        planeDim = 1;
        heightDim++;
        widthDim++;
        timeDim++;
      }

      /* sizes */
      i64 numPlanes = input.size(planeDim);
      i64 inputT = input.size(timeDim);
      i64 inputH = input.size(heightDim);
      i64 inputW = input.size(widthDim);

      TORCH_CHECK(outputT == gradOutput_.size(timeDim),
               "fractional_max_pool3d_backward_out(): gradOutput time unexpected");
      TORCH_CHECK(outputH == gradOutput_.size(heightDim),
               "fractional_max_pool3d_backward_out(): ",
               "gradOutput height unexpected");
      TORCH_CHECK(outputW == gradOutput_.size(widthDim),
               "fractional_max_pool3d_backward_out(): gradOutput width unexpected");

      /* get contiguous gradOutput */
      auto gradOutput = gradOutput_.contiguous();

      /* resize */
      gradInput.resize_as_(input);
      gradInput.zero_();

      /* backprop */
      AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "fractional_max_pool3d_backward_out_frame",
        [&]{
          fractional_max_pool3d_backward_out_frame<Scalar>(
            gradInput.data_ptr<Scalar>(),
            gradOutput.data_ptr<Scalar>(),
            indices.data_ptr<i64>(),
            numBatch, numPlanes,
            inputT, inputH, inputW,
            outputT, outputH, outputW
          );
        }
      );
        */
}

pub fn fractional_max_pool3d_out_cpu(
        input:          &Tensor,
        pool_size:      &[i32],
        output_size:    &[i32],
        random_samples: &Tensor,
        output:         &mut Tensor,
        indices:        &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            fractional_max_pool3d_out_cpu_template(
        output,
        indices,
        input,
        pool_size,
        output_size,
        randomSamples);
      return tuple<Tensor&, Tensor&>(output, indices);
        */
}

pub fn fractional_max_pool3d_cpu(
        input:          &Tensor,
        pool_size:      &[i32],
        output_size:    &[i32],
        random_samples: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            Tensor output = empty(output_size, input.options());
      Tensor indices = empty(output_size, kLong);
      fractional_max_pool3d_out_cpu_template(
        output,
        indices,
        input,
        pool_size,
        output_size,
        randomSamples);
      return tuple<Tensor, Tensor>(output, indices);
        */
}

pub fn fractional_max_pool3d_backward_out_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        pool_size:   &[i32],
        output_size: &[i32],
        indices:     &Tensor,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            fractional_max_pool3d_backward_out_cpu_template(
        input,
        gradOutput_,
        gradInput,
        output_size,
        pool_size,
        indices);
      return gradInput;
        */
}

pub fn fractional_max_pool3d_backward_cpu(
        grad_output: &Tensor,
        input:       &Tensor,
        pool_size:   &[i32],
        output_size: &[i32],
        indices:     &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor gradInput = empty({0}, input.options());
      fractional_max_pool3d_backward_out_cpu_template(
        input,
        gradOutput_,
        gradInput,
        output_size,
        pool_size,
        indices);
      return gradInput;
        */
}
