crate::ix!();

use crate::{
    Workspace,
    OperatorStorage,
    HIPContext,
    OperatorDef,
    MIOPENWrapper
};

                                                                                         
pub struct TensorDescriptors<T> {
    descs:  Vec<miopenTensorDescriptor_t>,
    phantom: PhantomData<T>,
}

impl<T> TensorDescriptors<T> {

    /*
      TensorDescriptors(
          size_t n,
          // dim and stride are not declared as const as opposed to cuDNN
          // since miopenSetTensorDescriptor doesn't take const arguments
          std::vector<int>& dim,
          std::vector<int>& stride);
    */
    
    #[inline] pub fn descs(&self) -> *const miopenTensorDescriptor_t {
        
        todo!();
        /*
            return descs_.data();
        */
    }
}

///---------------------------------
pub struct RecurrentBaseOp<T> {
    //USE_OPERATOR_FUNCTIONS(HIPContext);
    storage: OperatorStorage,
    context: HIPContext,

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
    phantom: PhantomData<T>,
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
    
///------------------------------------
pub struct RecurrentOp<T> {
    //USE_RECURRENT_BASE_FUNCTIONS
    base: RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

impl<T> RecurrentOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(operator_def, ws)
        */
    }
}

input_tags!{
    RecurrentOp {
        Input,
        HiddenInput,
        CellInput,
        Weight
    }
}

output_tags!{
    RecurrentOp {
        Output,
        HiddenOutput,
        CellOutput,
        RnnScratch,
        DropoutStates
    }
}

#[derive(PartialEq,Eq)]
pub enum RecurrentParamOpMode { SET_PARAM, GET_PARAM }

///-----------------------
pub struct RecurrentParamAccessOp<T,const mode: RecurrentParamOpMode> {
    //USE_RECURRENT_BASE_FUNCTIONS
    base: RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

impl<T,const mode: RecurrentParamOpMode> RecurrentParamAccessOp<T,mode> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(operator_def, ws)
        */
    }
}

///-----------------------
pub struct RecurrentGradientOp<T> {
    //USE_RECURRENT_BASE_FUNCTIONS
    base: RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

impl<T> RecurrentGradientOp<T> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(operator_def, ws)
        */
    }
}

input_tags!{
    RecurrentGradientOp {
        Input,
        HiddenInput,
        CellInput,
        Weight,
        RnnScratch,
        Output,
        GradOutput,
        GradHiddenOutput,
        GradCellOutput
    }
}

output_tags!{
    RecurrentGradientOp {
        GradInput,
        GradHiddenInput,
        GradCellInput,
        GradWeight,
        DropoutStates,
        RnnScratchOut
    }
}

