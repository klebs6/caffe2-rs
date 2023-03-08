crate::ix!();

pub struct OnnxWhileOpLocalScope {

    loop_ws:             *mut Workspace,

    /// owned by a workspace
    body_net:            *mut NetBase,
    iteration_var:       *mut Tensor,
    input_condition_var: *mut Tensor,
    condition_var:       *mut Tensor,
    lcd_tensors:         Vec<*mut Tensor>,
}

impl OnnxWhileOpLocalScope {
    
    pub fn new(
        loop_ws:      *mut Workspace,
        body_net_def: &NetDef,
        num_lcds:     usize) -> Self {
        todo!();
        /*
            : loop_ws_(loop_ws) 

          CAFFE_ENFORCE(loop_ws_, "Failed to initialize local loop workspace");

          // Create loop-carried deps in Workspace
          lcd_tensors_.clear();
          for (int i = 2; i < num_lcds + 2; ++i) {
            Blob* b = loop_ws_->CreateBlob(body_net_def.external_input(i));
            Tensor* t = BlobGetMutableTensor(b, Context::GetDeviceType());
            lcd_tensors_.push_back(t);
          }
          // First output is the iteration variable
          auto* iteration_var_blob =
              loop_ws_->CreateBlob(body_net_def.external_input(0));
          iteration_var_ =
              BlobGetMutableTensor(iteration_var_blob, Context::GetDeviceType());

          input_condition_var_ = BlobGetMutableTensor(
              loop_ws_->CreateBlob(body_net_def.external_input(1)),
              Context::GetDeviceType());

          auto* condition_var_blob =
              loop_ws_->CreateBlob(body_net_def.external_output(0));
          condition_var_ =
              BlobGetMutableTensor(condition_var_blob, Context::GetDeviceType());
          condition_var_->Resize(1);
          condition_var_->template mutable_data<bool>();

          body_net_ = loop_ws_->GetNet(body_net_def.name());
          if (!body_net_) {
            body_net_ = loop_ws_->CreateNet(body_net_def, true);
          }
          CAFFE_ENFORCE(body_net_, "Failed to initialize loop subnet");
        */
    }
    
    #[inline] pub fn net(&self) -> *mut NetBase {
        
        todo!();
        /*
            return body_net_;
        */
    }
    
    #[inline] pub fn workspace(&self) -> *mut Workspace {
        
        todo!();
        /*
            return loop_ws_;
        */
    }
    
    #[inline] pub fn iteration(&self) -> i64 {
        
        todo!();
        /*
            auto* iteration_var_ptr =
              iteration_var_->template mutable_data<int64_t>();
          return *iteration_var_ptr;
        */
    }
    
    #[inline] pub fn lcd_tensor(&mut self, idx: i32) -> *mut Tensor {
        
        todo!();
        /*
            return lcd_tensors_[idx];
        */
    }
    
    #[inline] pub fn set_iteration(&mut self, itr: i64)  {
        
        todo!();
        /*
            iteration_var_->Resize();
          auto* iteration_var_ptr =
              iteration_var_->template mutable_data<int64_t>();
          *iteration_var_ptr = itr;
        */
    }
    
    #[inline] pub fn set_input_condition<CondVarType>(&mut self, cond_value: bool)  {
        todo!();
        /*
            input_condition_var_->Resize(1);
          auto* input_condition_var_ptr =
              input_condition_var_->template mutable_data<CondVarType>();
          *input_condition_var_ptr = cond_value;
        */
    }
    
    #[inline] pub fn output_condition<CondVarType>(&self) -> bool {
        todo!();
        /*
            auto* condition_var_ptr =
              condition_var_->template mutable_data<CondVarType>();
          return *condition_var_ptr;
        */
    }
}
