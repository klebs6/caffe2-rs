crate::ix!();

/**
  | Although these seem generic, they make
  | subtle assumptions about the structure
  | of the graph that is 100% valid for NNModule
  | graphs but not any graph (such as data
  | being a unique_ptr).
  |
  */
pub struct inheritedFrom<T,U> {
    phantomA: PhantomData<T>,
    phantomB: PhantomData<U>,
    /*
  static constexpr bool value =
      std::is_base_of<U, T>::value && !std::is_same<U, T>::value;
    */
}

/**
  | This is just a way to fix issues when the
  | isa<> implementation can't automatically
  | downcast.
  |
  */
pub trait is_impl<N> { 

    #[inline] fn impl_(n: N) -> bool {
        
        todo!();
        /*
            return isa<T>(n->data());
        */
    }
}

impl<N> is_impl<N> for NeuralNetOperator {
    
    #[inline] fn impl_(n: N) -> bool {
        
        todo!();
        /*
            if (!isa<NeuralNetOperator>(n->data().get())) {
              return false;
            }
            auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
            return isa<T>(nno);
        */
    }
}

impl<N> is_impl<N> for NeuralNetData {
    
    #[inline] fn impl_(n: N) -> bool {
        todo!();

        /*
            if (!isa<NeuralNetData>(n->data().get())) {
              return false;
            }
            auto nno = dyn_cast<NeuralNetData>(n->data().get());
            return isa<T>(nno);
        */
    }
}

#[inline] pub fn is<T,U>(n: NodeRef<T,U>) -> bool {

    todo!();
    /*
        return is_impl<T, NodeRef>::impl(n);
    */
}

/**
  | This is just a way to fix issues when the
  | dyn_cast<> implementation can't automatically
  | downcast.
  |
  */
pub trait get_impl<N> {
    
    #[inline] fn impl_(n: N) -> *mut Self {
        
        todo!();
        /*
            return dyn_cast<T>(n->data().get());
        */
    }
}

impl<N> get_impl<N> for NeuralNetOperator {
    
    #[inline] fn impl_(n: N) -> *mut Self {

        todo!();
        /*
            if (!is<T>(n)) {
              assert(0 && "Cannot get type from node");
              return nullptr;
            }
            auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
            return dyn_cast<T>(nno);
        */
    }
}

impl<N> get_impl<N> for NeuralNetData {
    
    #[inline] fn impl_(n: N) -> *mut Self {

        todo!();
        /*
            if (!is<T>(n)) {
              assert(0 && "Cannot get type from node");
              return nullptr;
            }
            auto nno = dyn_cast<NeuralNetData>(n->data().get());
            return dyn_cast<T>(nno);
        */
    }
}

#[inline] pub fn get<T, N>(n: N) -> *mut T {

    todo!();
    /*
        return get_impl<T, N>::impl(n);
    */
}

