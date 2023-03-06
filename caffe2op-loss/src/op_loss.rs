crate::ix!();

/**
  | The *AveragedLoss* op takes a single
  | 1-D input tensor *input* and returns
  | a single output float value *output*.
  | The output represents the average of
  | the values in *input*. This op is commonly
  | used for averaging losses, hence the
  | name, however it does not exclusively
  | operate on losses.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc
  | 
  | AveragedLoss takes in the input and
  | produces the output loss value as the
  | average of the input.
  |
  */
pub struct AveragedLoss<T, Context> {
    base: SumElementsOp<T, Context>,

    phantom: PhantomData<T>,
}

num_inputs!{AveragedLoss, 1}

num_outputs!{AveragedLoss, 1}

inputs!{AveragedLoss, 
    0 => ("input", "The input data as Tensor")
}

outputs!{AveragedLoss, 
    0 => ("output", "The output tensor of size 1 containing the averaged value.")
}

scalar_type!{AveragedLoss, TensorProto::FLOAT}


impl<T, Context> AveragedLoss<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SumElementsOp<T, Context>(std::forward<Args>(args)..., true)
        */
    }
}

///----------------------------------
pub struct AveragedLossGradient<T, Context> {
    base: SumElementsGradientOp<T, Context>,
    phantom: PhantomData<T>,
}

num_inputs!{AveragedLossGradient, 2}

num_outputs!{AveragedLossGradient, 1}

impl<T, Context> AveragedLossGradient<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SumElementsGradientOp<T, Context>(std::forward<Args>(args)..., true)
        */
    }
}

register_cpu_operator!{AveragedLoss, AveragedLoss<f32, CPUContext>}

register_cpu_operator!{AveragedLossGradient, AveragedLossGradient<f32, CPUContext>}

#[test] fn averaged_loss_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "AveragedLoss",
        ["input"],
        ["output"],
    )

    workspace.FeedBlob("input", np.array([8, 10, 12]).astype(np.float32))
    print("input:\n", workspace.FetchBlob("input"))

    workspace.RunOperatorOnce(op)
    print("output: \n", workspace.FetchBlob("output"))


    input:
     [ 8. 10. 12.]
    output:
     10.0
    */
}


pub struct GetAveragedLossGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetAveragedLossGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "AveragedLossGradient", "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{AveragedLoss, GetAveragedLossGradient}
