crate::ix!();

pub struct NodeEqualityDefault<T> {
    phantom: PhantomData<T>,
}

impl<T> NodeEqualityDefault<T> {
    #[inline] pub fn equal(a: &T, b: &T) -> bool {
        todo!();
        /* return a->data() == b->data(); */
    }
}

pub type EqualityClassDefault<G: GraphType> = NodeEqualityDefault<<G as GraphType>::NodeRef>;
