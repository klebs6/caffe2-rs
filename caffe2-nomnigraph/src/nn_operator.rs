crate::ix!();

pub struct NeuralNetOperator {
    base: Instruction,

    kind:              NNKind,

    /// Mutable attribute, much like a type cast
    layout:            NNLayout,

    extra_annotation:  Box<Annotation>,
}

impl Default for NeuralNetOperator {
    
    fn default() -> Self {
        todo!();
        /*
            : Instruction(), kind_(NNKind::Undefined), layout_(NNLayout::Undefined
        */
    }
}

impl From<NNKind> for NeuralNetOperator {

    fn from(k: NNKind) -> Self {
    
        todo!();
        /*
            : Instruction(), kind_(K), layout_(NNLayout::Undefined)
        */
    }
}

impl Named for NeuralNetOperator {
    
    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            switch (getKind()) {
    #include "nomnigraph/Generated/OpNames.h"
        case NNKind::While:
          return "While";
        case NNKind::NNPhi:
          return "Phi";
        case NNKind::GenericOperator:
          return dyn_cast<GenericOperator>(this)->getName();
        default:
          return "Unknown";
      }
        */
    }
}

impl NeuralNetOperator {
    
    pub fn new_from_kind_opcode_layout(k: NNKind, i: Opcode, l: NNLayout) -> Self {

        todo!();
        /*
            : Instruction(I), kind_(K), layout_(L)
        */
    }
    
    pub fn new_from_kind_and_opcode(k: NNKind, i: Opcode) -> Self {
    
        todo!();
        /*
            : Instruction(I), kind_(K), layout_(NNLayout::Undefined)
        */
    }
    
    pub fn new_from_kind_and_layout(k: NNKind, l: NNLayout) -> Self {
    
        todo!();
        /*
            : Instruction(), kind_(K), layout_(L)
        */
    }
    
    #[inline] pub fn get_kind(&self) -> NNKind {
        
        todo!();
        /*
            return kind_;
        */
    }
    
    #[inline] pub fn set_layout(&mut self, l: NNLayout)  {
        
        todo!();
        /*
            layout_ = L;
        */
    }
    
    #[inline] pub fn get_layout(&self) -> NNLayout {
        
        todo!();
        /*
            return layout_;
        */
    }
    
    #[inline] pub fn set_annotation(
        &mut self, 
        extra_annotation: Box<Annotation>)
    {
        
        todo!();
        /*
            extraAnnotation_ = std::move(extraAnnotation);
        */
    }
    
    #[inline] pub fn get_annotation(&self) -> *const Annotation {
        
        todo!();
        /*
            return extraAnnotation_.get();
        */
    }
    
    #[inline] pub fn get_mutable_annotation(&mut self) -> *mut Annotation {
        
        todo!();
        /*
            return extraAnnotation_.get();
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Validate the inputs and outputs to this
      | operator.
      | 
      | \p inputs A vector of references to
      | 
      | NeuralNetData types that represent
      | the data being fed into the operator.
      | 
      | \p outputs A vector of references to
      | 
      | NeuralNetData types that represent
      | the data being outputted by the operator.
      | 
      | 
      | -----------
      | @return
      | 
      | true if the inputs and outputs are compatible
      | with the operator.
      |
      */
    #[inline] pub fn check_inputs_and_outputs(
        &mut self, 
        inputs:  Vec<*const NeuralNetData>, 
        outputs: Vec<*const NeuralNetData>) -> bool 
    {

        todo!();
        /*
            return true;
        */
    }
}

