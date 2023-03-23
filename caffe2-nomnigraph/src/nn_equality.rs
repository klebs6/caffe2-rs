crate::ix!();

pub struct NNEquality { }

impl NNEquality {
    
    #[inline] pub fn equal<T,U>(a: &NodeRef<T,U>, b: &NodeRef<T,U>) -> bool {
        
        todo!();
        /*
            if (
            !nom::repr::nn::is<nom::repr::NeuralNetOperator>(a) ||
            !nom::repr::nn::is<nom::repr::NeuralNetOperator>(b)) {
          return false;
        }
        auto a_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(a);
        auto b_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(b);

        bool sameKind = a_->getKind() == b_->getKind();
        if (sameKind && a_->getKind() == nom::repr::NeuralNetOperator::NNKind::GenericOperator) {
          return a_->getName() == b_->getName();
        }
        return sameKind;
        */
    }
}

