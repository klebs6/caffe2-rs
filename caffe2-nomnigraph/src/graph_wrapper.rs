crate::ix!();

pub struct GraphWrapper<T, U> {
    phantomA: PhantomData<T>,
    phantomB: PhantomData<U>,
}

impl<T,U> HasEdgeWrapper for GraphWrapper<T,U> {
    type EdgeWrapper = i32;//TODO where is the proper type for EdgeWrapper?
}

pub trait HasEdgeWrapper {
    type EdgeWrapper;
}
