crate::ix!();

/**
  | Given a packed dataset (packed by the
  | PackRecordsOp) and the `fields` argument
  | describing the datasets schema, return
  | the original dataset format. Number
  | of returned tensors is equal to the number
  | of fields in the `fields` argument.
  | 
  | The first input is the packed tensor
  | to be unpacked. Optionally, you can
  | provide prototype tensors to give the
  | expected shapes of the output tensors.
  | This is helpful when you expected to
  | unpack empty tensor, e.g., output of
  | a sampling process.
  |
  */
pub struct UnPackRecordsOp {
    storage: OperatorStorage,
    context: CPUContext,
    fields:  Vec<String>,
}

num_inputs!{UnPackRecords, (1,INT_MAX)}

num_outputs!{UnPackRecords, (1,INT_MAX)}

inputs!{UnPackRecords, 
    0 => ("packed_tensor", "The tensor to be unpacked")
}

args!{UnPackRecords, 
    0 => ("fields", "List of strings representing the string names in the format specified in the doc for CreateTreeCursor.")
}

impl UnPackRecordsOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            fields_(OperatorStorage::GetRepeatedArgument<std::string>("fields"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            size_t numRows = 0;
        Shared2DTensorVectorPtr data_ptr = nullptr;
        if (Input(0).IsType<SharedTensorVectorPtr>()) {
          numRows = Input(0).numel();
          CAFFE_ENFORCE_GE(numRows, 0);
          data_ptr = std::make_shared<Tensor2DVector>();
          data_ptr->reserve(numRows);

          const auto* inputs = Input(0).template data<SharedTensorVectorPtr>();
          for (int i = 0; i < numRows; i++) {
            data_ptr->emplace_back(*inputs[i]);
          }
        } else if (Input(0).IsType<Shared2DTensorVectorPtr>()) {
          CAFFE_ENFORCE_EQ(Input(0).numel(), 1);
          const auto* inputs = Input(0).template data<Shared2DTensorVectorPtr>();
          CAFFE_ENFORCE(inputs[0] != nullptr);
          data_ptr = inputs[0];
          numRows = inputs[0]->size();
          CAFFE_ENFORCE_GE(numRows, 0);
        } else {
          // input contains a single tensor
          CAFFE_ENFORCE_EQ(InputSize(), 1);
          CAFFE_ENFORCE_EQ(OutputSize(), 1);
          Output(0)->CopyFrom(Input(0));
          return true;
        }

        auto numTensors = OutputSize();

        // Precomputer the output sizes to avoid resizing
        std::vector<std::vector<int64_t>> outputDims(numTensors);
        std::vector<TypeMeta> metas(numTensors);

        CAFFE_ENFORCE(
            numRows > 0 || InputSize() > 1,
            "Unpacking empty record without shape will leave output blobs in "
            "undefined state.");

        if (InputSize() == 1) {
          getShapeAndMetaFromInput(data_ptr, outputDims, metas);
        } else {
          getShapeAndMetaFromPrototypeBlobs(outputDims, metas);
        }

        // inputs contains a single shared_ptr of vector<vector<caffe2::TensorCPU>>
        auto& tensors = *data_ptr;
        for (int i = 0; i < numRows; ++i) {
          for (int j = 0; j < tensors[i].size(); ++j) {
            const auto& input = tensors[i][j];

            // Checks to ensure that dimensions/sizes match
            CAFFE_ENFORCE_EQ(outputDims[j].size(), input.dim());
            CAFFE_ENFORCE(metas[j] == input.dtype());
            // We look from first dimension, because we concat on the first.
            for (int k = 1; k < input.dim(); ++k) {
              CAFFE_ENFORCE_EQ(input.sizes()[k], outputDims[j][k]);
            }

            outputDims[j][0] += input.size(0);
          }
        }

        // Resize to the final output size
        std::vector<void*> destinations(numTensors);
        for (int i = 0; i < numTensors; ++i) {
          Output(i)->Resize(outputDims[i]);
          destinations[i] = Output(i)->raw_mutable_data(metas[i]);
        }

        for (int i = 0; i < numRows; ++i) {
          for (int j = 0; j < numTensors; ++j) {
            const auto& input = tensors[i][j];

            context_.CopyItemsSameDevice(
                metas[j],
                input.numel(),
                input.raw_data() /* src */,
                destinations[j] /* dst */
            );

            destinations[j] =
                (char*)destinations[j] + input.numel() * input.itemsize();
          }
        }

        return true;
        */
    }
    
    #[inline] pub fn get_shape_and_meta_from_input(
        &mut self, 
        inputs:      &Shared2DTensorVectorPtr,
        output_dims: &mut Vec<Vec<i64>>,
        metas:       &mut Vec<TypeMeta>)
    {
        todo!();
        /*
            const auto& inputZero = inputs->at(0);

        const auto numTensors = inputZero.size();

        CAFFE_ENFORCE_EQ(numTensors, fields_.size());
        CAFFE_ENFORCE_EQ(numTensors, OutputSize());

        for (int i = 0; i < numTensors; ++i) {
          outputDims[i] = inputZero[i].sizes().vec();
          outputDims[i][0] = 0;
          metas[i] = inputZero[i].dtype();
        }
        */
    }
    
    #[inline] pub fn get_shape_and_meta_from_prototype_blobs(
        &mut self, 
        output_dims: &mut Vec<Vec<i64>>,
        metas:       &mut Vec<TypeMeta>)  
    {
        
        todo!();
        /*
            const auto numTensors = fields_.size();
        CAFFE_ENFORCE_EQ(numTensors, InputSize() - 1);
        CAFFE_ENFORCE_EQ(numTensors, OutputSize());
        for (int i = 0; i < numTensors; ++i) {
          const auto& input = Input(i + 1);
          outputDims[i] = input.sizes().vec();
          outputDims[i][0] = 0;
          metas[i] = input.dtype();
        }
        */
    }
}
