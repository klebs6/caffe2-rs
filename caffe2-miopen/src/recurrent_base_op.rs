crate::ix!();

#[macro_export] macro_rules! use_recurrent_base_functions {
    () => {
        /*
           USE_OPERATOR_FUNCTIONS(HIPContext);        
           using RecurrentBaseOp<T>::miopen_wrapper_;   
           using RecurrentBaseOp<T>::rnnDesc_;         
           using RecurrentBaseOp<T>::wDesc_;           
           using RecurrentBaseOp<T>::hxDesc_;          
           using RecurrentBaseOp<T>::cxDesc_;          
           using RecurrentBaseOp<T>::hyDesc_;          
           using RecurrentBaseOp<T>::cyDesc_;          
           using RecurrentBaseOp<T>::xDesc_;           
           using RecurrentBaseOp<T>::yDesc_;           
           using RecurrentBaseOp<T>::cachedInputDims_; 
           using RecurrentBaseOp<T>::reserveNbytes_;   
           using RecurrentBaseOp<T>::miopenWsNbytes_;   
           using RecurrentBaseOp<T>::initialize;
           */
    }
}

#[USE_OPERATOR_FUNCTIONS = "HIPContext"]
pub struct RecurrentBaseOp<T> {
    storage:            OperatorStorage,
    context:            HIPContext,
    miopen_wrapper:     MIOPENWrapper,
    rnn_desc:           miopenRNNDescriptor_t,
    w_desc:             miopenTensorDescriptor_t,
    hx_desc:            miopenTensorDescriptor_t,
    cx_desc:            miopenTensorDescriptor_t,
    hy_desc:            miopenTensorDescriptor_t,
    cy_desc:            miopenTensorDescriptor_t,
    x_desc:             Box<TensorDescriptors<T>>,
    y_desc:             Box<TensorDescriptors<T>>,
    cached_input_dims:  Vec<i64>,
    reserve_nbytes:     usize,
    miopen_ws_nbytes:   usize,
    phantom:            PhantomData<T>,
}

impl<T> RecurrentBaseOp<T> {

    /*TODO
      RecurrentBaseOp(const OperatorDef& operator_def, Workspace* ws);

     protected:
      void initialize(
          const Tensor& input,
          // If passed, reshapes to the appropriate size
          Tensor* output = nullptr,
          Tensor* hiddenOutput = nullptr,
          Tensor* cellOutput = nullptr);
    */
}
