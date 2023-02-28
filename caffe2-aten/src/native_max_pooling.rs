crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/MaxPooling.h]

// TODO(Heitor) Template by dimension
pub struct PoolingParams1D {

    /**
      | Number of batches
      |
      */
    NB: i64,


    /**
      | Number of channels
      |
      */
    NC: i64,


    /**
      | Input width
      |
      */
    IW: i64,


    /**
      | Output width
      |
      */
    OW: i64,


    /**
      | Kernel width
      |
      */
    KW: i64,


    /**
      | Column stride
      |
      */
    SJ: i64,


    /**
      | Column padding
      |
      */
    PJ: i64,


    /**
      | Column dilation
      |
      */
    DJ: i64,
}

impl PoolingParams1D {

    /**
      | Return index of input element for the
      | given kernel and output index
      |
      */
    #[inline] pub fn index(&self, kj: i64, oj: i64) -> i64 {
        
        todo!();
        /*
            return oj * SJ + kj * DJ - PJ;
        */
    }

    /**
      | Return index of first output within
      | bounds for this kernel index
      |
      */
    #[inline] pub fn valid_output_start(&self, kj: i64) -> i64 {
        
        todo!();
        /*
            i64 ij = index(kj, 0);;
        return ij < 0 ? divup(-ij, SJ) : 0;
        */
    }

    /**
      | Return index one past last output within
      | bounds for this kernel index
      |
      */
    #[inline] pub fn valid_output_end(&self, kj: i64) -> i64 {
        
        todo!();
        /*
            i64 ij = index(kj, OW - 1);
        return ij >= IW ? OW - divup(ij - (IW - 1), SJ) : OW;
        */
    }
}

lazy_static!{
    /*
    using pooling_fn = void (*)(Tensor&, const Tensor&, const PoolingParams1D&);
    */
}

declare_dispatch!{pooling_fn, max_pool1d_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/MaxPooling.cpp]

define_dispatch!{max_pool1d_stub}

pub fn max_pool1d_impl(
    self_:       &Tensor,
    kernel_size: &[i32],
    stride:      &[i32],
    padding:     &[i32],
    dilation:    &[i32],
    ceil_mode:   bool) -> Tensor {
    
    todo!();
        /*
            NoNamesGuard guard;

      TORCH_CHECK(
          self.dim() == 2 || self.dim() == 3,
          "max_pool1d() input tensor must have 2 or 3 dimensions but got ",
          self.dim());
      TORCH_CHECK(
          kernel_size.size() == 1,
          "max_pool1d() kernel_size must be an int or int list of size 1 but got size ",
          kernel_size.size());
      TORCH_CHECK(
          stride.size() == 0 || stride.size() == 1,
          "max_pool1d() stride must be None, an int or int list of size 1 but got size ",
          stride.size());
      TORCH_CHECK(
          padding.size() == 1,
          "max_pool1d() padding must be an int or int list of size 1 but got size ",
          padding.size());
      TORCH_CHECK(
          dilation.size() == 1,
          "max_pool1d() dilation must be an int or int list of size 1 but got size ",
          dilation.size());

      // If stride=None then set it to kernel_size
      if (stride.empty()) {
        stride = kernel_size;
      }

      const i64 NB = self.dim() == 3 ? self.size(-3) : 1;
      const i64 NC = self.size(-2);
      const i64 IW = self.size(-1);
      const i64 KW = kernel_size[0];
      const i64 SJ = stride[0];
      const i64 PJ = padding[0];
      const i64 DJ = dilation[0];

      TORCH_CHECK(
          KW > 0,
          "max_pool1d() kernel_size must be greater than zero, but got ",
          KW);
      TORCH_CHECK(
          SJ > 0, "max_pool1d() stride must be greater than zero, but got ", SJ);
      TORCH_CHECK(
          PJ >= 0, "max_pool1d() padding must be non-negative, but got ", PJ);
      TORCH_CHECK(
          PJ <= KW / 2,
          "max_pool1d() padding should be at most half of kernel size, but got padding=",
          PJ,
          " and kernel_size=",
          KW);
      TORCH_CHECK(
          DJ > 0, "max_pool1d() dilation must be greater than zero, but got ", DJ);

      const i64 OW = pooling_output_shape(IW, KW, PJ, SJ, DJ, ceil_mode);
      TORCH_CHECK(OW >= 0, "max_pool1d() Invalid computed output size: ", OW);
      Tensor output = empty({NB, NC, OW}, self.options());

      PoolingParams1D params{NB, NC, IW, OW, KW, SJ, PJ, DJ};
      max_pool1d_stub(self.device().type(), output, self, params);

      if (self.dim() == 2) {
        output.squeeze_(0);
      }

      guard.reset();
      namedinference::propagate_names(output, self);

      return output;
        */
}

pub fn max_pool1d(
        self_:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> Tensor {
    
    todo!();
        /*
            if (self.is_quantized()) {
        return quantized_max_pool1d(
            self, kernel_size, stride, padding, dilation, ceil_mode);
      }
      if ((self.requires_grad() && GradMode::is_enabled()) ||
          !self.device().is_cpu()) {
        // Needs indices for grad and with_indices defines CUDA dispatch
        return get<0>(max_pool1d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode));
      }
      return max_pool1d_impl(
          self, kernel_size, stride, padding, dilation, ceil_mode);
        */
}
