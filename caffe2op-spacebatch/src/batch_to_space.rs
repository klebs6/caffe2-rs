crate::ix!();

/**
  | Rearranges (permutes) data from batch
  | into blocks of spatial data, followed
  | by cropping.
  | 
  | This is the reverse transformation
  | of `SpaceToBatch`.
  | 
  | More specifically, this op outputs
  | a copy of the input tensor where values
  | from the batch dimension are moved in
  | spatial blocks to the height and width
  | dimensions, followed by cropping along
  | the height and width dimensions.
  | 
  | Only "NCHW" order is currently supported.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BatchToSpaceOp<Context> {
    base: SpaceBatchOpBase<Context>,
}

register_cpu_operator!{
    BatchToSpace, 
    BatchToSpaceOp<CPUContext>
}

num_inputs!{BatchToSpace, 1}

num_outputs!{BatchToSpace, 1}

inputs!{BatchToSpace, 
    0 => ("X",          "(*Tensor`<float>`*): input tensor (NCHW order)")
}

outputs!{BatchToSpace, 
    0 => ("Y",          "(*Tensor`<float>`*): output tensor (NCHW order)")
}

args!{BatchToSpace, 
    0 => ("pad",        "(*int*): exclusive axis that divides the first and second dimension of matrix `A` (default=0)"),
    1 => ("block_size", "(*int*): height/width of spatial blocks to be moved (default=2)"),
    2 => ("order",      "(*string*): order of dimensions of input and output blobs; only NCHW order is currently supported (default=NCHW)")
}

impl<Context> BatchToSpaceOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input = Input(0);
        auto* output = Output(0);
        const int batch = input.dim32(0);
        const int depth = input.dim32(1);
        const int height = input.dim32(2);
        const int width = input.dim32(3);

        const int output_batch = batch / this->block_size_ / this->block_size_;
        const int output_height =
            height * this->block_size_ - this->pad_b_ - this->pad_t_;
        const int output_width =
            width * this->block_size_ - this->pad_l_ - this->pad_r_;
        Output(0)->Resize(output_batch, depth, output_height, output_width);
        batchToSpace<Context>(
            input,
            this->pad_t_,
            this->pad_l_,
            this->block_size_,
            output,
            &context_);
        return true;
        */
    }
}

#[inline] pub fn batch_to_space<Context>(
    input:      &Tensor,
    pad_t:      i32,
    pad_l:      i32,
    block_size: i32,
    output:     *mut Tensor,
    context:    *mut Context)  {

    todo!();
    /*
        CAFFE_ENFORCE(input.dim() == 4);
      CAFFE_ENFORCE(output->dim() == 4);

      const int output_batch = output->dim32(0);
      const int output_depth = output->dim32(1);
      const int output_height = output->dim32(2);
      const int output_width = output->dim32(3);

      const int input_batch = input.dim32(0);
      const int input_depth = input.dim32(1);
      const int input_height = input.dim32(2);
      const int input_width = input.dim32(3);

      CAFFE_ENFORCE(input_depth == output_depth);
      for (int in_b = 0; in_b < input_batch; ++in_b) {
        const int out_b = in_b % output_batch;
        const int offset_w = (in_b / output_batch) % block_size;
        const int offset_h = (in_b / output_batch) / block_size;
        for (int d = 0; d < input_depth; ++d) {
          for (int in_h = 0; in_h < input_height; ++in_h) {
            const int out_h = in_h * block_size + offset_h - pad_t;
            for (int in_w = 0; in_w < input_width; ++in_w) {
              const int out_w = in_w * block_size + offset_w - pad_l;
              if (out_h >= 0 && out_w >= 0 && out_h < output_height &&
                  out_w < output_width) {
                const auto output_offset =
                    ((out_b * output_depth + d) * output_height + out_h) *
                        output_width +
                    out_w;
                const auto input_offset =
                    ((in_b * input_depth + d) * input_height + in_h) * input_width +
                    in_w;
                output->template mutable_data<float>()[output_offset] =
                    input.template data<float>()[input_offset];
              }
            }
          }
        }
      }
    */
}
