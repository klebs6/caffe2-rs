crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
    GradientMakerBase,
    StorageOrder,
    OperatorDef
};

#[inline] pub fn space_to_batch<Context>(
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

      for (int out_b = 0; out_b < output_batch; ++out_b) {
        const int in_b = out_b % input_batch;
        const int offset_w = (out_b / input_batch) % block_size;
        const int offset_h = (out_b / input_batch) / block_size;
        for (int d = 0; d < input_depth; ++d) {
          for (int out_h = 0; out_h < output_height; ++out_h) {
            const int in_h = out_h * block_size + offset_h - pad_t;
            for (int out_w = 0; out_w < output_width; ++out_w) {
              const int in_w = out_w * block_size + offset_w - pad_l;
              const auto output_offset =
                  ((out_b * output_depth + d) * output_height + out_h) *
                      output_width +
                  out_w;
              const auto input_offset =
                  ((in_b * input_depth + d) * input_height + in_h) * input_width +
                  in_w;
              if (in_h >= 0 && in_w >= 0 && in_h < input_height &&
                  in_w < input_width) {
                output->template mutable_data<float>()[output_offset] =
                    input.template data<float>()[input_offset];
              } else {
                output->template mutable_data<float>()[output_offset] = 0.0;
              }
            }
          }
        }
      }
    */
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


///-----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpaceBatchOpBase<Context> {
    storage: OperatorStorage,
    context: Context,

    pad:         i32,
    pad_t:       i32,
    pad_l:       i32,
    pad_b:       i32,
    pad_r:       i32,
    block_size:  i32,
    order:       StorageOrder,
}

impl<Context> SpaceBatchOpBase<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            pad_(this->template GetSingleArgument<int>("pad", 0)),
            pad_t_(this->template GetSingleArgument<int>("pad_t", pad_)),
            pad_l_(this->template GetSingleArgument<int>("pad", pad_)),
            pad_b_(this->template GetSingleArgument<int>("pad", pad_)),
            pad_r_(this->template GetSingleArgument<int>("pad", pad_)),
            block_size_(this->template GetSingleArgument<int>("block_size", 2)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))) 

        CAFFE_ENFORCE(order_ == StorageOrder::NCHW);
        */
    }
}

///-----------------------------------

#[test] fn space_to_batch_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SpaceToBatch",
        ["X"],
        ["Y"],
        pad=2,
        block_size=3
    )

    workspace.FeedBlob("X", np.random.rand(1,3,5,5).astype(np.float32))
    print("X.shape:", workspace.FetchBlob("X").shape)
    workspace.RunOperatorOnce(op)
    print("Y.shape:", workspace.FetchBlob("Y").shape)


    X.shape: (1, 3, 5, 5)
    Y.shape: (9, 3, 3, 3)

    */
}

/**
  | Zero-pads and then rearranges (permutes)
  | blocks of spatial data into batch. More
  | specifically, this op outputs a copy
  | of the input tensor where values from
  | the height and width dimensions are
  | moved to the batch dimension. After
  | the zero-padding is according to the
  | `pad` argument, both height and width
  | of the input must be divisible by the
  | `block_size`. Only "NCHW" order is
  | currently supported.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SpaceToBatchOp<Context> {

    base: SpaceBatchOpBase<Context>,
}

register_cpu_operator!{SpaceToBatch, SpaceToBatchOp<CPUContext>}

num_inputs!{SpaceToBatch, 1}

num_outputs!{SpaceToBatch, 1}

inputs!{SpaceToBatch, 
    0 => ("X", "(*Tensor`<float>`*): input tensor (NCHW order)")
}

outputs!{SpaceToBatch, 
    0 => ("Y", "(*Tensor`<float>`*): output tensor (NCHW order)")
}

args!{SpaceToBatch, 
    0 => ("pad", "(*int*): exclusive axis that divides the first and second dimension of matrix `A` (default=0)"),
    1 => ("block_size", "(*int*): height/width of spatial blocks to be moved (default=2)"),
    2 => ("order", "(*string*): order of dimensions of input and output blobs; only NCHW order is currently supported (default=NCHW)")
}

impl<Context> SpaceToBatchOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input = Input(0);
        auto* output = Output(0);
        const int batch = input.dim32(0);
        const int depth = input.dim32(1);
        const int height = this->pad_b_ + this->pad_t_ + input.dim32(2);
        const int width = this->pad_l_ + this->pad_r_ + input.dim32(3);
        CAFFE_ENFORCE(
            height % this->block_size_ == 0,
            "Height: ",
            height,
            ", block size: ",
            this->block_size_);
        CAFFE_ENFORCE(width % this->block_size_ == 0);

        const int output_batch = batch * this->block_size_ * this->block_size_;
        const int output_height = height / this->block_size_;
        const int output_width = width / this->block_size_;
        Output(0)->Resize(output_batch, depth, output_height, output_width);

        spaceToBatch<Context>(
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

register_cpu_operator!{BatchToSpace, BatchToSpaceOp<CPUContext>}

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

#[test] fn batch_to_space_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "BatchToSpace",
        ["X"],
        ["Y"],
        pad=3
    )

    workspace.FeedBlob("X", np.random.rand(10,3,32,32).astype(np.float32))
    print("X.shape:", workspace.FetchBlob("X").shape)
    workspace.RunOperatorOnce(op)
    print("Y.shape:", workspace.FetchBlob("Y").shape)

    X.shape: (10, 3, 32, 32)
    Y.shape: (2, 3, 58, 58)
    */
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

pub struct GetSpaceToBatchGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetSpaceToBatchGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchToSpace", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

pub struct GetBatchToSpaceGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetBatchToSpaceGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SpaceToBatch", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{SpaceToBatch, GetSpaceToBatchGradient}

register_gradient!{BatchToSpace, GetBatchToSpaceGradient}
