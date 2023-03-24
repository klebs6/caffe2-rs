crate::ix!();

pub type PredicateFn = fn(opdef: &OperatorDef) -> bool;

pub struct VisitorContext<T,U> {
    infos:          HashMap<NodeRef<T,U>,GroupAnnotation>,
    frontier:       HashSet<NodeRef<T,U>>,
    current_group:  Vec<NodeRef<T,U>>,
    group:          i32,  //{0};
    find_supported: bool, //{true};
    predicate:      PredicateFn,
}

impl<T,U> VisitorContext<T,U> {

    pub fn new(predicate: PredicateFn) -> Self {
        Self {
            infos:          HashMap::<NodeRef<T,U>,GroupAnnotation>::new(),
            frontier:       HashSet::<NodeRef<T,U>>::new(),
            current_group:  vec![],
            group:          0,
            find_supported: true,
            predicate:      predicate,
        }
    }
}

