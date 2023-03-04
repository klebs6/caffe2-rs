crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HeatmapMaxKeypointOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    should_output_softmax: bool,// = false;

    /**
      | Input: heatmaps [size x size], boxes [x0,
      | y0, x1, y1]
      |
      | Output: keypoints (#rois, 4, #keypoints)
      */
    phantom: PhantomData<T>,
}

impl<T,Context> HeatmapMaxKeypointOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            should_output_softmax_(this->template GetSingleArgument<bool>(
                "should_output_softmax",
                false))
        */
    }
}

register_cpu_operator!{
    HeatmapMaxKeypoint,
    HeatmapMaxKeypointOp<f32, CPUContext>
}

num_inputs!{HeatmapMaxKeypoint, 2}

num_outputs!{HeatmapMaxKeypoint, 1}

should_not_do_gradient!{HeatmapMaxKeypoint}

