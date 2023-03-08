crate::ix!();

impl<T, const mode: RecurrentParamOpMode> RecurrentParamAccessOp<T, mode> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
    
        todo!();
        /*
            initialize(Input(0));

      if (mode == SET_PARAM) {
        size_t paramsSize;
        CUDNN_ENFORCE(cudnnGetRNNParamsSize(
            cudnn_wrapper_.inline_cudnn_handle(),
            rnnDesc_,
            xDesc_->descs()[0],
            &paramsSize,
            cudnnTypeWrapper<T>::type));

        CAFFE_ENFORCE_EQ(
            paramsSize / 4, Input(1).numel(), "Incorrect weight initialization");
      }

      int layer = OperatorStorage::GetSingleArgument<int>("layer", 0);
      std::string param_type =
          OperatorStorage::GetSingleArgument<string>("param_type", "");
      std::string input_type =
          OperatorStorage::GetSingleArgument<string>("input_type", "");

      // Mapping to CUDNN constants
      std::map<string, int> weight_constants = {{"input_gate_w", 0},
                                                {"forget_gate_w", 1},
                                                {"cell_w", 2},
                                                {"output_gate_w", 3}};
      std::map<string, int> bias_constants = {{"input_gate_b", 0},
                                              {"forget_gate_b", 1},
                                              {"cell_b", 2},
                                              {"output_gate_b", 3}};
      if (bias_constants.find(param_type) != bias_constants.end()) {
        int param_id = bias_constants[param_type] + 4 * (input_type == "recurrent");

        cudnnFilterDescriptor_t biasDesc;
        CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&biasDesc));
        void* bias;

        CUDNN_ENFORCE(cudnnGetRNNLinLayerBiasParams(
            cudnn_wrapper_.inline_cudnn_handle(),
            rnnDesc_,
            layer,
            xDesc_->descs()[0],
            wDesc_,
            Input(1).template data<T>(),
            param_id, // Forget gate bias for recurrent input
            biasDesc,
            &bias));
        int numBiasDims;
        std::vector<int> biasDims(3);
        cudnnDataType_t dt;
        cudnnTensorFormat_t tf;
        // For some reason, the Cudnn Bias tensor is 3 dimensional
        CUDNN_ENFORCE(cudnnGetFilterNdDescriptor(
            biasDesc, 3, &dt, &tf, &numBiasDims, biasDims.data()));
        CAFFE_ENFORCE_EQ(numBiasDims, 3);

        if (mode == SET_PARAM) {
          CAFFE_ENFORCE_EQ(
              biasDims[0] * biasDims[1] * biasDims[2], Input(2).numel());
          this->context_.template CopySameDevice<T>(
              biasDims[0] * biasDims[1] * biasDims[2],
              Input(2).template data<T>(),
              static_cast<T*>(bias));
        } else {
          Output(0)->Resize(biasDims);
          this->context_.template CopySameDevice<T>(
              biasDims[0] * biasDims[1] * biasDims[2],
              static_cast<T*>(bias),
              Output(0)->template mutable_data<T>());
        }
      } else if (weight_constants.find(param_type) != weight_constants.end()) {
        int param_id =
            weight_constants[param_type] + 4 * (input_type == "recurrent");
        cudnnFilterDescriptor_t matrixParamDesc;
        CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&matrixParamDesc));
        void* pmatrix;
        CUDNN_ENFORCE(cudnnGetRNNLinLayerMatrixParams(
            cudnn_wrapper_.inline_cudnn_handle(),
            rnnDesc_,
            layer,
            xDesc_->descs()[0],
            wDesc_,
            Input(1).template data<T>(),
            param_id, // Forget gate bias for recurrent input
            matrixParamDesc,
            &pmatrix));
        int numDims;
        std::vector<int> matDims(3);
        cudnnDataType_t dt;
        cudnnTensorFormat_t tf;

        CUDNN_ENFORCE(cudnnGetFilterNdDescriptor(
            matrixParamDesc, 3, &dt, &tf, &numDims, matDims.data()));
        CAFFE_ENFORCE_EQ(numDims, 3);
        if (mode == SET_PARAM) {
          CAFFE_ENFORCE_EQ(matDims[0] * matDims[1] * matDims[2], Input(2).numel());
          this->context_.template CopySameDevice<T>(
              matDims[0] * matDims[1] * matDims[2],
              Input(2).template data<T>(),
              static_cast<T*>(pmatrix));
        } else {
          Output(0)->Resize(matDims);
          this->context_.template CopySameDevice<T>(
              matDims[0] * matDims[1] * matDims[2],
              static_cast<T*>(pmatrix),
              Output(0)->template mutable_data<T>());
        }
      } else {
        CAFFE_ENFORCE(false, "Unknown param type:", param_type);
      }

      return true;
        */
    }
}
