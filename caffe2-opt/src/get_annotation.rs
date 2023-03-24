crate::ix!();

/**
  | If the annotation doesn't exist, attempt
  | to add it
  |
  */
#[inline] pub fn get_or_add_caffe2_annotation<T,U>(instr_node: &mut NodeRef<T,U>) -> *mut Caffe2Annotation<T,U> {
    
    todo!();
    /*
        auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
      auto* annotation = nnOp->getMutableAnnotation();
      if (!annotation) {
        auto new_annot = std::make_unique<Caffe2Annotation>();
        new_annot->setOperatorDef(convertToOperatorDef(instrNode));
        nnOp->setAnnotation(std::move(new_annot));
        annotation = nnOp->getMutableAnnotation();
      }
      CAFFE_ENFORCE(isa<Caffe2Annotation>(annotation));
      auto c2_annotation = dyn_cast<Caffe2Annotation>(annotation);
      return c2_annotation;
    */
}
