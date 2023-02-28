crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/MaxUnpooling.cpp]

pub fn max_unpooling2d_forward_out_cpu_frame<Scalar>(
    output:  &mut Tensor,
    input:   &Tensor,
    indices: &Tensor,
    oheight: i64,
    owidth:  i64) -> Tensor {

    todo!();
        /*
            i64 numBatch = 1;
      i64 dimc = 0;
      i64 dimh = 1;
      i64 dimw = 2;
      if (input.ndimension() == 4) {
        numBatch = input.size(0);
        dimc++;
        dimh++;
        dimw++;
      }
      i64 numChannels = input.size(dimc);
      i64 inputHeight = input.size(dimh);
      i64 inputWidth = input.size(dimw);

      auto* rawInput = input.data_ptr<Scalar>();
      auto* rawIndices = indices.data_ptr<i64>();
      auto* rawOutput = output.data_ptr<Scalar>();

      internal::lazy_init_num_threads();

      for (i64 n = 0; n < numBatch; n++) {
        i64 nOutputOffset = n * numChannels * owidth * oheight;
        i64 nInputOffset = n * numChannels * inputWidth * inputHeight;
        i64 k = 0;
        bool has_error = false;
        i64 error_index = 0;
    #pragma omp parallel for private(k)
        for (k = 0; k < numChannels; k++) {
          i64 finalOutputOffset = nOutputOffset + k * owidth * oheight;
          i64 finalInputOffset = nInputOffset + k * inputWidth * inputHeight;
          Scalar* output_p_k = rawOutput + finalOutputOffset;
          Scalar* input_p_k = rawInput + finalInputOffset;
          i64* ind_p_k = rawIndices + finalInputOffset;

          i64 maxp;
          for (i64 i = 0; i < inputHeight; i++) {
            for (i64 j = 0; j < inputWidth; j++) {
              maxp = ind_p_k[i * inputWidth + j];
              if (maxp < 0 || maxp >= owidth * oheight) {
    #pragma omp critical
                {
                  has_error = true;
                  error_index = maxp;
                }
              } else {
                output_p_k[maxp] = input_p_k[i * inputWidth + j];
              }
            }
          }
        }
        if (has_error) {
          AT_ERROR(
              "Found an invalid max index: ",
              error_index,
              " (output volumes are of size ",
              oheight,
              "x",
              owidth);
        }
      }
      return output;
        */
}

pub fn max_unpooling2d_forward_out_cpu(
    self_:       &Tensor,
    indices:     &Tensor,
    output_size: &[i32],
    output:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto oheight = output_size[0];
      auto owidth = output_size[1];
      TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
      TORCH_CHECK(
          indices_.scalar_type() == ScalarType::Long,
          "elements in indices should be type int64");
      TORCH_CHECK(
          output_size.size() == 2,
          "There should be exactly two elements (height, width) in output_size");
      TORCH_CHECK(
          (self_.ndimension() == 3 || self_.ndimension() == 4),
          "Input to max_unpooling2d should be a 3d or 4d Tensor");
      TORCH_CHECK(
          self_.sizes() == indices_.sizes(),
          "Shape of indices should match shape of input");

      TORCH_CHECK(self_.numel() > 0, "Input must be non-empty");

      auto self = self_.contiguous();
      auto indices = indices_.contiguous();

      if (self.ndimension() == 3) {
        i64 numChannels = self.size(0);
        output.resize_({numChannels, oheight, owidth});
      } else {
        i64 numBatch = self.size(0);
        i64 numChannels = self.size(1);
        output.resize_({numBatch, numChannels, oheight, owidth});
      }
      output.zero_();

      AT_DISPATCH_FLOATING_TYPES(
          self.scalar_type(), "max_unpooling2d_forward_out_cpu_frame", ([&] {
            max_unpooling2d_forward_out_cpu_frame<Scalar>(
                output, self, indices, oheight, owidth);
          }));
      return output;
        */
}

pub fn max_unpooling2d_forward_cpu(
        self_:       &Tensor,
        indices:     &Tensor,
        output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            auto output = empty({0}, self.options());
      native::max_unpooling2d_forward_out_cpu(self, indices, output_size, output);
      return output;
        */
}

pub fn max_unpooling3d_forward_out_cpu_frame<Scalar>(
        output:  &mut Tensor,
        input:   &Tensor,
        indices: &Tensor,
        ot:      i64,
        oh:      i64,
        ow:      i64) -> Tensor {

    todo!();
        /*
            i64 nBatch = 1;
      i64 dimw = 3;
      i64 dimh = 2;
      i64 dimt = 1;

      if (input.ndimension() == 5) {
        nBatch = input.size(0);
        dimw++;
        dimh++;
        dimt++;
      }

      i64 nSlices = input.size(dimt - 1);
      i64 iT = input.size(dimt);
      i64 iH = input.size(dimh);
      i64 iW = input.size(dimw);

      Scalar* input_data = input.data_ptr<Scalar>();
      Scalar* output_data = output.data_ptr<Scalar>();
      i64* indices_data = indices.data_ptr<i64>();

      internal::lazy_init_num_threads();

      for (i64 p = 0; p < nBatch; p++) {
        i64 inputOffset = p * nSlices * iT * iW * iH;
        i64 outputOffset = p * nSlices * oT * oW * oH;
        i64 k = 0;
        bool has_error = false;
        int error_index = 0;
    #pragma omp parallel for private(k)
        for (k = 0; k < nSlices; k++) {
          i64 finalInputOffset = inputOffset + k * iT * iW * iH;
          i64 finalOutputOffset = outputOffset + k * oT * oW * oH;

          Scalar* output_p_k = output_data + finalOutputOffset;
          Scalar* input_p_k = input_data + finalInputOffset;
          i64* ind_p_k = indices_data + finalInputOffset;
          int maxp;
          for (i64 t = 0; t < iT; t++) {
            for (i64 i = 0; i < iH; i++) {
              for (i64 j = 0; j < iW; j++) {
                i64 index = t * iH * iW + i * iW + j;
                maxp = ind_p_k[index];
                if (maxp < 0 || maxp >= oT * oW * oH) {
    #pragma omp critical
                  {
                    has_error = true;
                    error_index = maxp;
                  }
                } else {
                  output_p_k[maxp] = input_p_k[index];
                }
              }
            }
          }
          if (has_error) {
            AT_ERROR(
                "found an invalid max index ",
                error_index,
                " (output volumes are of size ",
                oT,
                "x",
                oH,
                "x",
                oW);
          }
        }
      }
      return output;
        */
}

pub fn max_unpooling3d_shape_check(
        input:       &Tensor,
        grad_output: &Tensor,
        indices:     &Tensor,
        output_size: &[i32],
        stride:      &[i32],
        padding:     &[i32])  {
    
    todo!();
        /*
            i64 oT = output_size[0];
      i64 oH = output_size[1];
      i64 oW = output_size[2];
      TORCH_CHECK(
          indices.scalar_type() == ScalarType::Long,
          "elements in indices should be type int64");
      TORCH_CHECK(
          (input.ndimension() == 4 || input.ndimension() == 5),
          "Input to max_unpooling3d should be a 4d or 5d Tensor",
          input.sizes());
      TORCH_CHECK(
          output_size.size() == 3,
          "There should be exactly three elements (depth, height, width) in output_size");
      TORCH_CHECK(
          stride.size() == 3,
          "There should be exactly three elements (depth, height, width) in stride");
      TORCH_CHECK(
          padding.size() == 3,
          "There should be exactly three elements (depth, height, width) in padding");
      TORCH_CHECK(
          input.sizes() == indices.sizes(),
          "Shape of indices should match shape of input");

      TORCH_CHECK(input.numel() > 0, "Input must be non-empty");

      TORCH_CHECK(
          stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
          "strides should be greater than zero, but got stride: ",
          stride);

      int dimw = 3;
      int dimh = 2;
      int dimt = 1;
      int dimn = 0;

      if (input.ndimension() == 5) {
        dimw++;
        dimh++;
        dimt++;
        dimn++;
      }

      int nslices = input.size(dimn);

      if (gradOutput.defined()) {
        if (oT != gradOutput.size(dimt) || oH != gradOutput.size(dimh) ||
            oW != gradOutput.size(dimw)) {
          AT_ERROR(
              "Inconsistent gradOutput size. oT= ",
              oT,
              ", oH= ",
              oH,
              ", oW= ",
              oW,
              ". gradOutput: ",
              gradOutput.size(dimt),
              "x",
              gradOutput.size(dimh),
              "x",
              gradOutput.size(dimw));
        }
        TORCH_CHECK(
            gradOutput.ndimension() == input.ndimension() &&
                gradOutput.size(dimn) == nslices,
            "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
      }
        */
}

pub fn max_unpooling3d_forward_out_cpu(
        self_:       &Tensor,
        indices:     &Tensor,
        output_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        output:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
      i64 oT = output_size[0];
      i64 oH = output_size[1];
      i64 oW = output_size[2];

      auto self = self_.contiguous();
      auto indices = indices_.contiguous();

      max_unpooling3d_shape_check(
          self_, Tensor(), indices_, output_size, stride, padding);

      if (self_.ndimension() == 5) {
        output.resize_({self.size(0), self.size(1), oT, oH, oW});
      } else {
        output.resize_({self.size(0), oT, oH, oW});
      }
      output.zero_();

      AT_DISPATCH_FLOATING_TYPES(
          self.scalar_type(), "max_unpooling3d_forward_out_cpu_frame", ([&] {
            max_unpooling3d_forward_out_cpu_frame<Scalar>(
                output,
                self,
                indices,
                oT,
                oH,
                oW);
          }));
      return output;
        */
}

pub fn max_unpooling3d_forward_cpu(
        self_:       &Tensor,
        indices:     &Tensor,
        output_size: &[i32],
        stride:      &[i32],
        padding:     &[i32]) -> Tensor {
    
    todo!();
        /*
            auto output = empty({0}, self.options());
      native::max_unpooling3d_forward_out_cpu(
          self, indices, output_size, stride, padding, output);
      return output;
        */
}

pub fn max_unpooling2d_backward_out_cpu_frame<Scalar>(
        grad_input_p:  *mut Scalar,
        grad_output_p: *mut Scalar,
        ind_p:         *mut i64,
        nslices:       i64,
        iheight:       i64,
        iwidth:        i64,
        oheight:       i64,
        owidth:        i64)  {

    todo!();
        /*
            bool has_error = false;
      i64 error_index = 0;
      i64 k = 0;

      internal::lazy_init_num_threads();
    #pragma omp parallel for private(k)
      for (k = 0; k < nslices; k++) {
        Scalar* gradInput_p_k = gradInput_p + k * iwidth * iheight;
        Scalar* gradOutput_p_k = gradOutput_p + k * owidth * oheight;
        i64* ind_p_k = ind_p + k * iwidth * iheight;

        i64 i, j;
        i64 maxp;

        for (i = 0; i < iheight; i++) {
          for (j = 0; j < iwidth; j++) {
            maxp = ind_p_k[i * iwidth + j]; /* retrieve position of max */
            if (maxp < 0 || maxp >= owidth * oheight) {
    #pragma omp critical
              {
                has_error = true;
                error_index = maxp;
              }
            }
            gradInput_p_k[i * iwidth + j] =
                gradOutput_p_k[maxp]; /* update gradient */
          }
        }
      }
      if (has_error) {
        AT_ERROR(
            "invalid max index ",
            error_index,
            ", owidth= ",
            owidth,
            ", oheight= ",
            oheight);
      }
        */
}

pub fn max_unpooling2d_backward_out_cpu(
        grad_output: &Tensor,
        self_:       &Tensor,
        indices:     &Tensor,
        output_size: &[i32],
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
      i64 oheight = output_size[0];
      i64 owidth = output_size[1];
      int dimw = 2;
      int dimh = 1;
      int nbatch = 1;
      int nslices;
      int iheight;
      int iwidth;
      TORCH_CHECK(
          indices_.scalar_type() == ScalarType::Long,
          "elements in indices should be type int64");
      TORCH_CHECK(
          self.sizes() == indices_.sizes(), "Input shape must match indices shape");

      TORCH_CHECK(output_size.size() == 2, "Output size must be 2");

      /* get contiguous gradOutput and indices */
      auto grad_output = grad_output_.contiguous();
      auto indices = indices_.contiguous();

      /* resize */
      grad_input.resize_as_(self);
      grad_input.zero_();

      if (self.ndimension() == 4) {
        nbatch = self.size(0);
        dimw++;
        dimh++;
      }

      /* sizes */
      nslices = self.size(dimh - 1);
      iheight = self.size(dimh);
      iwidth = self.size(dimw);

      if (owidth != grad_output.size(dimw) || oheight != grad_output.size(dimh)) {
        AT_ERROR(
            "Inconsistent gradOutput size. output height = ",
            oheight,
            ", output width = ",
            owidth,
            ", gradOutput: ",
            grad_output.size(dimh),
            "x",
            grad_output.size(dimw));
      }
      AT_DISPATCH_FLOATING_TYPES(
          self.scalar_type(), "max_unpooling2d_backward_out_cpu_frame", ([&] {
            int p;
            for (p = 0; p < nbatch; p++) {
              auto inputOffset = p * nslices * iheight * iwidth;
              auto outputOffset = p * nslices * oheight * owidth;
              max_unpooling2d_backward_out_cpu_frame<Scalar>(
                  grad_input.data_ptr<Scalar>() + inputOffset,
                  grad_output.data_ptr<Scalar>() + outputOffset,
                  indices.data_ptr<i64>() + inputOffset,
                  nslices,
                  iheight,
                  iwidth,
                  oheight,
                  owidth);
            }
          }));
      return grad_input;
        */
}

pub fn max_unpooling2d_backward_cpu(
        grad_output: &Tensor,
        self_:       &Tensor,
        indices:     &Tensor,
        output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            auto grad_input = empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      native::max_unpooling2d_backward_out_cpu(
          grad_output, self, indices, output_size, grad_input);
      return grad_input;
        */
}

pub fn max_unpooling3d_backward_out_cpu_frame<Scalar>(
        grad_input_p:  *mut Scalar,
        grad_output_p: *mut Scalar,
        ind_p:         *mut i64,
        nslices:       i64,
        it:            i64,
        ih:            i64,
        iw:            i64,
        ot:            i64,
        oh:            i64,
        ow:            i64)  {

    todo!();
        /*
            i64 k = 0;
      bool has_error = false;
      int error_index = 0;

      internal::lazy_init_num_threads();

    #pragma omp parallel for private(k)
      for (k = 0; k < nslices; k++) {
        Scalar* gradInput_p_k = gradInput_p + k * iT * iH * iW;
        Scalar* gradOutput_p_k = gradOutput_p + k * oT * oH * oW;
        i64* ind_p_k = ind_p + k * iT * iH * iW;

        i64 t, i, j, index;
        i64 maxp;
        for (t = 0; t < iT; t++) {
          for (i = 0; i < iH; i++) {
            for (j = 0; j < iW; j++) {
              index = t * iH * iW + i * iW + j;
              maxp = ind_p_k[index]; /* retrieve position of max */
              if (maxp < 0 || maxp >= oT * oH * oW) {
    #pragma omp critical
                {
                  has_error = true;
                  error_index = maxp;
                }
              }
              gradInput_p_k[index] = gradOutput_p_k[maxp]; /* update gradient */
            }
          }
        }
      }
      if (has_error) {
        AT_ERROR(
            "invalid max index ",
            error_index,
            ", oT= ",
            oT,
            ", oW= ",
            oW,
            ",oH= ",
            oH);
      }
        */
}

pub fn max_unpooling3d_backward_out_cpu(
        grad_output: &Tensor,
        self_:       &Tensor,
        indices:     &Tensor,
        output_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
      auto oT = output_size[0];
      auto oH = output_size[1];
      auto oW = output_size[2];
      int dimw = 3;
      int dimh = 2;
      int dimt = 1;
      int nbatch = 1;
      int nslices;
      int iT;
      int iH;
      int iW;

      max_unpooling3d_shape_check(
          self, grad_output_, indices_, output_size, stride, padding);

      // TODO (from THNN): check gradOutput shape
      /* get contiguous gradOutput */
      auto grad_output = grad_output_.contiguous();
      auto indices = indices_.contiguous();

      /* resize */
      grad_input.resize_as_(self);
      grad_input.zero_();
      if (self.ndimension() == 5) {
        nbatch = self.size(0);
        dimt++;
        dimw++;
        dimh++;
      }

      /* sizes */
      nslices = self.size(dimt - 1);
      iT = self.size(dimt);
      iH = self.size(dimh);
      iW = self.size(dimw);

      /* backprop */
      AT_DISPATCH_FLOATING_TYPES(
          self.scalar_type(), "max_unpooling3d_backward_out_cpu_frame", ([&] {
            int p;
            for (p = 0; p < nbatch; p++) {
              int inputOffset = p * nslices * iT * iH * iW;
              int outputOffset = p * nslices * oT * oH * oW;
              max_unpooling3d_backward_out_cpu_frame<Scalar>(
                  grad_input.data_ptr<Scalar>() + inputOffset,
                  grad_output.data_ptr<Scalar>() + outputOffset,
                  indices.data_ptr<i64>() + inputOffset,
                  nslices,
                  iT,
                  iH,
                  iW,
                  oT,
                  oH,
                  oW);
            }
          }));
      return grad_input;
        */
}

pub fn max_unpooling3d_backward_cpu(
        grad_output: &Tensor,
        self_:       &Tensor,
        indices:     &Tensor,
        output_size: &[i32],
        stride:      &[i32],
        padding:     &[i32]) -> Tensor {
    
    todo!();
        /*
            auto grad_input = empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      native::max_unpooling3d_backward_out_cpu(
          grad_output, self, indices, output_size, stride, padding, grad_input);
      return grad_input;
        */
}
