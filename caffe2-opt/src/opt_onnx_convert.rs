crate::ix!();

use nalgebra::coordinates::X;

/**
  | TORCH_API nom::repr::NNModule convertToNNModule(caffe2::NetDef
  | &net, std::unordered_map<std::string,
  | nom::repr::NNGraph::NodeRef>* blobMapOut
  | = nullptr);
  | 
  | TORCH_API caffe2::NetDef convertToOnnxProto(nom::repr::NNModule&);
  | 
  | TORCH_API std::unique_ptr<nom::repr::NeuralNetOperator>
  | convertToOperatorDef(caffe2::OperatorDef
  | op);
  |
  */
pub struct OnnxAnnotation {
    base: Annotation,

    device:  String, // = "";
    op_def:  *mut OperatorDef, // = nullptr;
}

impl Default for OnnxAnnotation {
    
    fn default() -> Self {
        todo!();
        /*
            : Annotation(AnnotationKind::Onnx
        */
    }
}

impl OnnxAnnotation {
    
    pub fn new(device: String) -> Self {
    
        todo!();
        /*
            : Annotation(AnnotationKind::Onnx), Device(device)
        */
    }
    
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
    
    #[inline] pub fn set_operator_def(&mut self, op_def: *mut OperatorDef)  {
        
        todo!();
        /*
            OpDef = opDef;
        */
    }
    
    #[inline] pub fn get_operator_def(&self) -> *const OperatorDef {
        
        todo!();
        /*
            assert(OpDef && "OperatorDef was never set.  Use OnnxAnnotation::setOperatorDef.");
        return OpDef;
        */
    }
    
    #[inline] pub fn get_mutable_operator_def(&mut self) -> *mut OperatorDef {
        
        todo!();
        /*
            assert(OpDef && "OperatorDef was never set.  Use OnnxAnnotation::setOperatorDef.");
        return OpDef;
        */
    }
    
    #[inline] pub fn classof(a: *const Annotation) -> bool {
        
        todo!();
        /*
            return A->getKind() == AnnotationKind::Onnx;
        */
    }
}
