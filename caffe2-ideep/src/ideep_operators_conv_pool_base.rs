crate::ix!();

use crate::{
    IDEEPTensorDims,
    IDEEPTensor,
    ConvPoolOpBase,
    Workspace,
    OperatorDef,
    IDEEPContext
};

pub struct IDEEPConvPoolOpBase {
    base: ConvPoolOpBase<IDEEPContext>,

}

impl IDEEPConvPoolOpBase {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<IDEEPContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn input(&mut self, index: i32) -> &IDEEPTensor {
        
        todo!();
        /*
            return OperatorStorage::template Input<ideep::tensor>(index);
        */
    }
    
    #[inline] pub fn output(&mut self, index: i32) -> *mut IDEEPTensor {
        
        todo!();
        /*
            return OperatorStorage::template Output<ideep::tensor>(index);
        */
    }
    
    #[inline] pub fn pad_tl(&self) -> IDEEPTensorDims {
        
        todo!();
        /*
            return {pad_t(), pad_l()};
        */
    }
    
    #[inline] pub fn pad_br(&self) -> IDEEPTensorDims {
        
        todo!();
        /*
            return {pad_b(), pad_r()};
        */
    }
    
    #[inline] pub fn calc_output_dims(
        &mut self, 
        input: &IDEEPTensor,
        output_channel: i32) -> IDEEPTensorDims 
    {
        todo!();
        /*
            CAFFE_ENFORCE_GT(input.get_size(), 0);
        std::vector<int> output_dims;
        const auto input_dims = input.get_dims();
        std::vector<std::int64_t> input_Tdims(
            input_dims.cbegin(), input_dims.cend());
        InferOutputSize(
            input_Tdims,
            output_channel,
            StorageOrder::NCHW, //order_,
            global_pooling_,
            legacy_pad_,
            dilation_,
            stride_,
            &kernel_,
            &pads_,
            &output_dims);
        return {output_dims.begin(), output_dims.end()};
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (!global_pooling_) {
          for (int dim = 0; dim < kernel_.size(); ++dim) {
            CAFFE_ENFORCE_GT(kernel_[dim], 0);
          }
        }

        try {
          return RunOnDeviceWithOrderNCHW();
        } catch (ideep::error& e) {
          LOG(ERROR) << "IDEEP error:" << e.message;
          throw;
        }
        */
    }
}

#[macro_export] macro_rules! use_ideep_conv_pool_base_functions {
    () => {
        todo!();
        /*
        USE_OPERATOR_BASE_FUNCTIONS;                           \
            /* using override */ using IDEEPConvPoolOpBase::Input; \
            /* using override */ using IDEEPConvPoolOpBase::Output;
            */
    }
}
