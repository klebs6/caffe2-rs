crate::ix!();

pub trait GraphType {
    type NodeRef;
    type EdgeRef;
    type SubgraphType;
}
