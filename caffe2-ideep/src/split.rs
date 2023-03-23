crate::ix!();

#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPSplitOp {
    base:        IDEEPOperator,
    axis:        i32,
    add_axis:    i32,
    axis_offset: Vec<i32>,
}

register_ideep_operator!{Split, IDEEPSplitOp}

input_tags!{
    IDEEPSplitOp {
        Input,
        AxisInfo
    }
}

impl IDEEPSplitOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            axis_offset_(OperatorStorage::GetRepeatedArgument<int>("split")) 

        CAFFE_ENFORCE(
          !(OperatorStorage::HasArgument("axis") && OperatorStorage::HasArgument("order")),
            "You shouldn't specify both the dim to split, and the order "
            "in the case of 4-D images.");
        if (OperatorStorage::HasArgument("axis")) {
          axis_ = OperatorStorage::GetSingleArgument<int>("axis", -1);
          // only exists for computing the gradient of a Concat with 'add_axis'
          add_axis_ = OperatorStorage::GetSingleArgument<int>("add_axis", 0);
        } else {
          axis_ = 1;
          add_axis_ = 0;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input = Input(INPUT);
        int canonical_axis = canonical_axis_index_(axis_, input.ndims());
        const int input_channels = input.get_dim(canonical_axis);
        vector<int> axis_vdata(OutputSize(), 0);
        if (InputSize() == 2) {
          // We obtain split from the input tensor.
          CAFFE_ENFORCE_EQ(
              axis_offset_.size(),
              0,
              "If you set split with an input blob, do not pass in "
              "split in the argument.");
          auto& axis_info = OperatorStorage::Input<Tensor>(AXIS_INFO, CPU);
          CAFFE_ENFORCE_EQ(axis_info.numel(), OutputSize());
          auto* axis_data = axis_info.template data<int>();
          axis_vdata.assign(axis_data, axis_data + OutputSize());
        } else if (axis_offset_.size() == 0) {
          CAFFE_ENFORCE_EQ(
              input_channels % OutputSize(),
              0,
              "If you did not specify split explicitly, the number of "
              "input channels should be divisible by the output size.");
          axis_vdata.assign(OutputSize(), input_channels / OutputSize());
        } else {
          // We obtain split from the parameters.
          CAFFE_ENFORCE_EQ(
              axis_offset_.size(),
              OutputSize(),
              "The number of splits specified should be equal to the "
              "number of outputs.");
          axis_vdata = axis_offset_;
        }

        CAFFE_ENFORCE_EQ(
            add_axis_ ? OutputSize()
                      : std::accumulate(
                        axis_vdata.data(), axis_vdata.data() + OutputSize(), 0),
            input_channels,
            "Sum of split dimensions do not match: should be ",
            input_channels);

        auto iten_vector = ideep::spliter::compute(
            input, axis_vdata, canonical_axis, add_axis_);
        CAFFE_ENFORCE_EQ(
            iten_vector.size(),
            OutputSize(),
            "Output size does not match: should be ",
            OutputSize());

        for (int i = 0; i < OutputSize(); i++) {
          auto* output = Output(i);
          *output = iten_vector[i];
        }

        return true;
        */
    }
}
