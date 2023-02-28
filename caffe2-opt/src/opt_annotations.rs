crate::ix!();

use crate::{
    NodeRef,
    Annotation,
    OperatorDef,
    DeviceOption,
};

pub enum ParallelizationScheme {
    none,
    split_by_batch,
    split_by_length,
    shard,
    shard_by_number
}

pub struct Caffe2Annotation<T,U> {

    base:                    Annotation,

    device:                  String,
    op_def:                  OperatorDef,
    op_def_exists:           bool, // default = false

    /// Distributed annotations
    device_type:             i32, // = caffe2::DeviceTypeProto::PROTO_CPU;

    parallelization_scheme:  ParallelizationScheme, // = ParallelizationScheme::none;

    /// = -1;
    parallelization:         i32,

    /// = nullptr;
    key_node:                NodeRef<T,U>,

    /// = nullptr;
    length_node:             NodeRef<T,U>,

    component_levels:        Vec<String>,
}

impl<T,U> Default for Caffe2Annotation<T,U> {
    
    fn default() -> Self {
        todo!();
        /*
            : Annotation(AnnotationKind::Caffe2
        */
    }
}

impl<T,U> Caffe2Annotation<T,U> {
    
    pub fn new(device: String) -> Self {
    
        todo!();
        /*
            : Annotation(AnnotationKind::Caffe2), Device(device)
        */
    }
    
    #[inline] pub fn set_operator_def(&mut self, op_def: &OperatorDef)  {
        
        todo!();
        /*
            OpDef = opDef;
      OpDefExists = true;
        */
    }
    
    #[inline] pub fn has_operator_def(&self) -> bool {
        
        todo!();
        /*
            return OpDefExists;
        */
    }
    
    #[inline] pub fn get_operator_def(&self) -> &OperatorDef {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          OpDefExists,
          "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
      return OpDef;
        */
    }
    
    #[inline] pub fn get_mutable_operator_def(&mut self) -> *mut OperatorDef {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          OpDefExists,
          "OperatorDef was never set.  Use Caffe2Annotation::setOperatorDef.");
      return &OpDef;
        */
    }
    
    /// Distributed annotations
    #[inline] pub fn set_device_option(&mut self, dev_opt: &DeviceOption)  {
        
        todo!();
        /*
            *OpDef.mutable_device_option() = devOpt;
        */
    }
    
    #[inline] pub fn has_device_option(&self) -> bool {
        
        todo!();
        /*
            return OpDef.has_device_option();
        */
    }
    
    #[inline] pub fn get_device_option(&self) -> &DeviceOption {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          hasDeviceOption(),
          "DeviceOption was never set.  Use Caffe2Annotation::setDeviceOption.");
      return OpDef.device_option();
        */
    }
    
    #[inline] pub fn get_mutable_device_option(&mut self) -> *mut DeviceOption {
        
        todo!();
        /*
            CAFFE_ENFORCE(
          hasDeviceOption(),
          "DeviceOption was never set.  Use Caffe2Annotation::setDeviceOption.");
      return OpDef.mutable_device_option();
        */
    }
    
    // Distributed annotations
    #[inline] pub fn set_device(&mut self, device: String)  {
        
        todo!();
        /*
            Device = device;
        */
    }
    
    #[inline] pub fn get_device(&self) -> String {
        
        todo!();
        /*
            return Device;
        */
    }
    
    #[inline] pub fn set_device_type(&mut self, device: i32)  {
        
        todo!();
        /*
            DeviceType = device;
        */
    }
    
    #[inline] pub fn get_device_type(&self) -> i32 {
        
        todo!();
        /*
            return DeviceType;
        */
    }
    
    #[inline] pub fn set_parallelization(&mut self, s: ParallelizationScheme, num: i32)  {
        
        todo!();
        /*
            parallelization_scheme_ = s;
      parallelization_ = num;
        */
    }
    
    #[inline] pub fn get_parallelization_scheme(&self) -> ParallelizationScheme {
        
        todo!();
        /*
            return parallelization_scheme_;
        */
    }
    
    #[inline] pub fn get_parallelization(&self) -> i32 {
        
        todo!();
        /*
            return parallelization_;
        */
    }
    
    #[inline] pub fn set_key_node(&mut self, n: NodeRef<T,U>)  {
        
        todo!();
        /*
            key_node_ = n;
        */
    }
    
    #[inline] pub fn get_key_node(&self) -> &NodeRef<T,U> {
        
        todo!();
        /*
            CAFFE_ENFORCE(key_node_, "No key node has been annotated");
      return key_node_;
        */
    }
    
    #[inline] pub fn set_length_node(&mut self, n: NodeRef<T,U>)  {
        
        todo!();
        /*
            length_node_ = n;
        */
    }
    
    #[inline] pub fn get_length_node(&self) -> &NodeRef<T,U> {
        
        todo!();
        /*
            CAFFE_ENFORCE(length_node_, "No length node has been annotated");
      return length_node_;
        */
    }
    
    #[inline] pub fn set_component_levels(&mut self, components: Vec<String>)  {
        
        todo!();
        /*
            component_levels_ = std::move(components);
        */
    }
    
    #[inline] pub fn get_component_levels(&self) -> Vec<String> {
        
        todo!();
        /*
            return component_levels_;
        */
    }
    
    #[inline] pub fn classof(&mut self, a: *const Annotation) -> bool {
        
        todo!();
        /*
            return A->getKind() == AnnotationKind::Caffe2;
        */
    }
}
