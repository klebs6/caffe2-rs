crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/include/conv_utils.h]

#[inline] pub fn compute_output_dimension(

    // Input dimension
    input_dim:      usize,

    // Input padding
    pad_dim:        usize,

    // Adjustment to the output dimension
    adjustment_dim: usize,

    // Kernel dimension
    kernel_dim:     usize,

    // Dilation dimension
    dilation_dim:   usize,

    // Stride or subsampling dimension
    stride_dim:     usize,

    // Transposed convolution
    transpose:      bool) -> usize {

    todo!();
        /*
            kernel_dim = (kernel_dim - 1) * dilation_dim + 1;  // Effective kernel dim
      if (transpose) {
        return stride_dim * (input_dim - 1) + adjustment_dim + kernel_dim - pad_dim;
      } else {
        return (input_dim + pad_dim - kernel_dim) / stride_dim + 1;
      }
        */
}

pub struct ConvParam {

    /**
      | kernel width, height
      |
      */
    kernel_dims:           [u32; 2],


    /**
      | subsampling width, height
      |
      */
    stride_dims:           [u32; 2],


    /**
      | dilation width, height
      |
      */
    dilation:              [u32; 2],


    /**
      | input padding top, left, bottom, right
      |
      */
    padding:               [u32; 4],


    /**
      | output adjustment
      |
      */
    adjustment_dims:       [u32; 2],

    groups:                u32,
    input_channels:        usize,
    output_channels:       usize,
    transpose:             bool,

    /**
      | The following are derived parameters
      | kernel type based on input params
      |
      */
    ukernel_type:          PyTorchQnnpUKernelType,

    group_input_channels:  usize,
    group_output_channels: usize,
    per_channel:           bool,

}

impl ConvParam {

    /**
      | -----------
      | @brief
      | 
      | Constructor for initializing the convolution/deconvolution
      | parameters.
      |
      */
    pub fn new(
        kernel:          [u32; 2],
        stride:          [u32; 2],
        dilation:        [u32; 2],
        padding:         [u32; 4],
        adjustment:      [u32; 2],
        groups:          u32,
        input_channels:  usize,
        output_channels: usize,
        transpose:       bool,
        is_per_channel:  bool) -> Self {
    
        todo!();
        /*


            : kernel_dims(kernel_),
            stride_dims(stride_),
            dilation(dilation_),
            padding(padding_),
            adjustment_dims(adjustment_),
            groups(groups_),
            input_channels(input_channels_),
            output_channels(output_channels_),
            transpose(transpose_),
            per_channel(is_per_channel_) 
        const u32 kernel_width = kernel_dims[0];
        const u32 kernel_height = kernel_dims[1];

        const u32 input_padding_top = padding[0];
        const u32 input_padding_left = padding[1];
        const u32 input_padding_bottom = padding[2];
        const u32 input_padding_right = padding[3];

        const char* _name;
        if (transpose) {
          _name = "deconvolution\0";
        } else {
          _name = "convolution\0";
        }

        if (groups == 0) {
          pytorch_qnnp_log_error(
              "failed to create %s with groups equal to zero.", _name);
          assert("Failed to initialize QNNPACK conv_param_t struct.");
        }

        if (input_channels % groups != 0 || output_channels % groups != 0) {
          pytorch_qnnp_log_error(
              "failed to create %s: input and output channels must be divisible by"
              " groups.", _name);
          assert("Failed to initialize QNNPACK conv_param_t struct.");
        }

        group_input_channels = input_channels / groups;
        group_output_channels = output_channels / groups;

        if (kernel_width == 0 || kernel_height == 0) {
          pytorch_qnnp_log_error(
              "failed to create %s with %" PRIu32 "x%" PRIu32
              " kernel: kernel dimensions must be non-zero",
              _name,
              kernel_width,
              kernel_height);
          assert("Failed to initialize QNNPACK conv_param_t struct.");
        }

        if (stride_dims[0] == 0 || stride_dims[1] == 0) {
          pytorch_qnnp_log_error(
              "failed to create %s with %" PRIu32 "x%" PRIu32
              " subsampling: "
              "subsampling dimensions must be non-zero",
              _name,
              stride_dims[0],
              stride_dims[1]);
          assert("Failed to initialize QNNPACK conv_param_t struct.");
        }

        if (dilation[0] == 0 || dilation[1] == 0) {
          pytorch_qnnp_log_error(
              "failed to create %s with %" PRIu32 "x%" PRIu32
              " dilation: "
              "dilation dimensions must be non-zero",
              _name,
              dilation[0],
              dilation[1]);
          assert("Failed to initialize QNNPACK conv_param_t struct.");
        }

        if (stride_dims[1] > kernel_height) {
          pytorch_qnnp_log_info(
              "inefficiency in %s with %" PRIu32 "x%" PRIu32 " kernel and %"
              PRIu32 "x%" PRIu32 " subsampling: "
              "height subsampling is greater than kernel height; subsampling should"
              " be performed before the %s",
              _name,
              kernel_width,
              kernel_height,
              stride_dims[0],
              stride_dims[1],
              _name);
        }

        if (stride_dims[0] > kernel_width) {
          pytorch_qnnp_log_info(
              "inefficiency in %s with %" PRIu32 "x%" PRIu32 " kernel and %"
              PRIu32 "x%" PRIu32 " subsampling: "
              "width subsampling is greater than kernel width; subsampling should"
              " be performed before the %s",
              _name,
              kernel_width,
              kernel_height,
              stride_dims[0],
              stride_dims[1],
              _name);
        }

        if (input_padding_top >= kernel_height) {
          pytorch_qnnp_log_info(
              "inefficiency in %s with %" PRIu32 "x%" PRIu32 " kernel and %"
              PRIu32 "+%" PRIu32 " height padding: "
              "input top padding is greater or equal to kernel height",
              _name,
              kernel_width,
              kernel_height,
              input_padding_top,
              input_padding_bottom);
        }

        if (input_padding_bottom >= kernel_height) {
          pytorch_qnnp_log_info(
              "inefficiency in %s with %" PRIu32 "x%" PRIu32 " kernel and %"
              PRIu32 "+%" PRIu32 " height padding: "
              "input bottom padding is greater or equal to kernel height",
              _name,
              kernel_width,
              kernel_height,
              input_padding_top,
              input_padding_bottom);
        }

        if (input_padding_right >= kernel_width) {
          pytorch_qnnp_log_info(
              "inefficiency in %s with %" PRIu32 "x%" PRIu32 " kernel and %"
              PRIu32 "+%" PRIu32 " width padding: "
              "input right padding is greater or equal to kernel width",
              _name,
              kernel_width,
              kernel_height,
              input_padding_left,
              input_padding_right);
        }

        if (input_padding_left >= kernel_width) {
          pytorch_qnnp_log_info(
              "inefficiency in %s with %" PRIu32 "x%" PRIu32 " kernel and %"
              PRIu32 "+%" PRIu32 " width padding: "
              "input left padding is greater or equal to kernel width",
              _name,
              kernel_width,
              kernel_height,
              input_padding_left,
              input_padding_right);
        }

        const usize kernel_size = kernel_height * kernel_width;
        if (transpose) {
          ukernel_type = pytorch_qnnp_ukernel_type_conv;
        } else {
          ukernel_type = pytorch_qnnp_ukernel_type_none;
          const bool any_padding = (input_padding_left | input_padding_top
              | input_padding_right | input_padding_bottom) != 0;

          if ((kernel_size == 9 || kernel_size == 25) &&
              group_input_channels == 1 && group_output_channels == 1 && groups > 1) {
            ukernel_type = pytorch_qnnp_ukernel_type_dwconv;
          } else if (kernel_size == 1 && stride_dims[1] == 1 && stride_dims[0] == 1 && !any_padding) {
            ukernel_type = group_input_channels >= SIZE_MAX ? pytorch_qnnp_ukernel_type_xzp_gemm : pytorch_qnnp_ukernel_type_gemm;
          } else {
            ukernel_type = pytorch_qnnp_ukernel_type_conv;
          }
        }

        if (per_channel && ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
          pytorch_qnnp_log_error(
              "Per channel quantized weights are not supported for XZP kernels");
          assert("Failed to initialize QNNPACK conv_param_t struct.");
        }
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Computes the output dimensions given
      | a 2D input.
      |
      */
    pub fn compute_output_dims(&self, input_dims: [usize; 2]) -> [usize; 2] {
        
        todo!();
        /*
            array<usize, 2> output_dims;
        output_dims[0] = compute_output_dimension(input_dims[0],  // width
                                                  padding[1] + padding[3],
                                                  adjustment_dims[0],
                                                  kernel_dims[0],
                                                  dilation[0],
                                                  stride_dims[0],
                                                  transpose);
        output_dims[1] = compute_output_dimension(input_dims[1],  // height
                                                  padding[0] + padding[2],
                                                  adjustment_dims[1],
                                                  kernel_dims[1],
                                                  dilation[1],
                                                  stride_dims[1],
                                                  transpose);
        return output_dims;
        */
    }
}
