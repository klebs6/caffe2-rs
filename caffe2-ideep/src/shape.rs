crate::ix!();

/**
  | RecordShapeOp records the shape of
  | the input tensor to a vector of int.
  | 
  | You mostly don't need this operator
  | explicitly, and it is mostly used in
  | the autodiff process.
  |
  */
#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_OPERATOR_FUNCTIONS]
pub struct IDEEPShapeOp {
    base: IDEEPOperator,
    axes: Vec<i32>,
}

input_tags!{
    IDEEPShapeOp {
        Data
    }
}

output_tags!{
    IDEEPShapeOp {
        Output
    }
}

impl IDEEPShapeOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPOperator(operator_def, ws),
            axes_(OperatorStorage ::GetRepeatedArgument<int>("axes"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int numDims = 0;
        int numAxes = axes_.size();
        vector<int64_t> dims;
        const char* data_dims = nullptr;
        auto* output = OperatorStorage::Output<Tensor>(OUTPUT, CPU);

        if (OperatorStorage::InputBlob(DATA).template IsType<itensor>()) {
          auto& data = Input(DATA);
          numDims = data.ndims();
          auto idims = data.get_dims();
          dims.assign(idims.begin(), idims.end());
          data_dims = reinterpret_cast<const char*>(dims.data());
        } else {
          auto& data = OperatorStorage::Input<Tensor>(DATA, CPU);
          numDims = data.dim();
          data_dims = reinterpret_cast<const char*>(data.sizes().data());
        }

        if (numAxes == 0) {
          output->Resize(numDims);
          int64_t* output_data = output->template mutable_data<int64_t>();
          context_.CopyBytesSameDevice(
              numDims * sizeof(int64_t), data_dims, output_data);
          return true;
        }

        output->Resize(numAxes);
        auto out = reinterpret_cast<char*>(output->template mutable_data<int64_t>());
        for (int i = 0; i < numAxes; i++) {
          auto axis = axes_[i];
          CAFFE_ENFORCE_LT(axis, numDims, "Axis out of range");
          CAFFE_ENFORCE_GE(axis, 0, "Each axis should be non-negative");
          context_.CopyBytesSameDevice(
              sizeof(int64_t), data_dims + axis * sizeof(int64_t), out);
          out += sizeof(int64_t);
        }

        return true;
        */
    }
}

