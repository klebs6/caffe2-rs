crate::ix!();

pub enum AnnotationKind { 
    Generic, 
    Caffe2 
}

/**
  | \brief Annotations allow for generic
  | manipulation of neural network operations.
  | The base class contains a saved void* pointer
  | for external use.  Derived classes add richer
  | semantics to the annotation and it is
  | encouraged to use them.
  */
pub struct Annotation {
    kind:  AnnotationKind,
}

impl Annotation {
    
    pub fn new(kind: AnnotationKind) -> Self {
    
        todo!();
        /*
            : kind_(kind)
        */
    }
    
    #[inline] pub fn get_kind(&self) -> AnnotationKind {
        
        todo!();
        /*
            return kind_;
        */
    }
}

impl Default for Annotation {
    
    fn default() -> Self {
        todo!();
        /*
            : kind_(AnnotationKind::Generic
        */
    }
}

