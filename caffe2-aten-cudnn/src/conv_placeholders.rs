crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cudnn/ConvPlaceholders.cpp]

// ---------------------------------------------------------------------
//
// Placeholder operators
//
// ---------------------------------------------------------------------

// See Note [ATen preprocessor philosophy]
#[cfg(not(AT_CUDNN_ENABLED))]
pub mod cudnn_disabled {

    use super::*;

    pub fn cudnn_convolution(
            input:         &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution: ATen not compiled with cuDNN support");
            */
    }

    pub fn cudnn_convolution_backward_input(
            input_size:    &[i32],
            grad_output:   &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_backward_input: ATen not compiled with cuDNN support");
            */
    }


    pub fn cudnn_convolution_backward_weight(
            weight_size:   &[i32],
            grad_output:   &Tensor,
            input:         &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_backward_weight: ATen not compiled with cuDNN support");
            */
    }

    pub fn cudnn_convolution_backward(
            input:         &Tensor,
            grad_output:   &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool,
            output_mask:   [bool; 2]) -> (Tensor,Tensor) {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_backward: ATen not compiled with cuDNN support");
            */
    }


    pub fn cudnn_convolution_transpose(
            input:          &Tensor,
            weight:         &Tensor,
            padding:        &[i32],
            output_padding: &[i32],
            stride:         &[i32],
            dilation:       &[i32],
            groups:         i64,
            benchmark:      bool,
            deterministic:  bool,
            allow_tf32:     bool) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_transpose: ATen not compiled with cuDNN support");
            */
    }

    pub fn cudnn_convolution_transpose_backward_input(
            grad_output:   &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
            */
    }

    pub fn cudnn_convolution_transpose_backward_weight(
            weight_size:   &[i32],
            grad_output:   &Tensor,
            input:         &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_transpose_backward_weight: ATen not compiled with cuDNN support");
            */
    }

    pub fn cudnn_convolution_transpose_backward(
            input:          &Tensor,
            grad_output:    &Tensor,
            weight:         &Tensor,
            padding:        &[i32],
            output_padding: &[i32],
            stride:         &[i32],
            dilation:       &[i32],
            groups:         i64,
            benchmark:      bool,
            deterministic:  bool,
            allow_tf32:     bool,
            output_mask:    [bool; 2]) -> (Tensor,Tensor) {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
            */
    }

    pub fn raw_cudnn_convolution_forward_out(
            output:        &Tensor,
            input:         &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool)  {
        
        todo!();
            /*
                AT_ERROR("raw_cudnn_convolution_forward_out: ATen not compiled with cuDNN support");
            */
    }

    pub fn raw_cudnn_convolution_backward_input_out(
            grad_input:    &Tensor,
            grad_output:   &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool)  {
        
        todo!();
            /*
                AT_ERROR("raw_cudnn_convolution_backward_input_out: ATen not compiled with cuDNN support");
            */
    }

    pub fn raw_cudnn_convolution_backward_weight_out(
            grad_weight:   &Tensor,
            grad_output:   &Tensor,
            input:         &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            benchmark:     bool,
            deterministic: bool,
            allow_tf32:    bool)  {
        
        todo!();
            /*
                AT_ERROR("raw_cudnn_convolution_backward_weight_out: ATen not compiled with cuDNN support");
            */
    }

    pub fn cudnn_convolution_relu(
            input_t:  &Tensor,
            weight_t: &Tensor,
            bias_t:   &Option<Tensor>,
            stride:   &[i32],
            padding:  &[i32],
            dilation: &[i32],
            groups:   i64) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_relu: ATen not compiled with cuDNN support");
            */
    }

    pub fn cudnn_convolution_add_relu(
            input_t:  &Tensor,
            weight_t: &Tensor,
            z_t:      &Tensor,
            alpha:    &Option<Scalar>,
            bias_t:   &Option<Tensor>,
            stride:   &[i32],
            padding:  &[i32],
            dilation: &[i32],
            groups:   i64) -> Tensor {
        
        todo!();
            /*
                AT_ERROR("cudnn_convolution_add_relu: ATen not compiled with cuDNN support");
            */
    }
}

// ---------------------------------------------------------------------
//
// Deprecated operators
//
// ---------------------------------------------------------------------

/**
  | TODO (@zasdfgbnm): this is here only
  | for compatibility, remove this in the
  | future
  |
  */
pub fn cudnn_convolution_deprecated(
        input:         &Tensor,
        weight:        &Tensor,
        bias_opt:      &Option<Tensor>,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      auto output = cudnn_convolution(input, weight, padding, stride, dilation, groups, benchmark, deterministic);
      if (bias.defined()) {
        output = output + reshape_bias(input.dim(), bias);
      }
      return output;
        */
}

/**
  | TODO (@zasdfgbnm): this is here only
  | for compatibility, remove this in the
  | future
  |
  */
pub fn cudnn_convolution_deprecated2(
        input_t:       &Tensor,
        weight_t:      &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            return cudnn_convolution(input_t, weight_t, padding, stride, dilation, groups, benchmark, deterministic, globalContext().allowTF32CuDNN());
        */
}

/**
  | TODO (@zasdfgbnm): this is here only
  | for compatibility, remove this in the
  | future
  |
  */
pub fn cudnn_convolution_transpose_deprecated(
        input:          &Tensor,
        weight:         &Tensor,
        bias_opt:       &Option<Tensor>,
        padding:        &[i32],
        output_padding: &[i32],
        stride:         &[i32],
        dilation:       &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      auto output = cudnn_convolution_transpose(input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      if (bias.defined()) {
        output = output + reshape_bias(input.dim(), bias);
      }
      return output;
        */
}

/**
  | TODO (@zasdfgbnm): this is here only
  | for compatibility, remove this in the
  | future
  |
  */
pub fn cudnn_convolution_transpose_deprecated2(
        input_t:        &Tensor,
        weight_t:       &Tensor,
        padding:        &[i32],
        output_padding: &[i32],
        stride:         &[i32],
        dilation:       &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool) -> Tensor {
    
    todo!();
        /*
            return cudnn_convolution_transpose(input_t, weight_t, padding, output_padding, stride, dilation, groups, benchmark, deterministic, globalContext().allowTF32CuDNN());
        */
}
