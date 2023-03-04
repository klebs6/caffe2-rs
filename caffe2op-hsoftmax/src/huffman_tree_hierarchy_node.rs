/*!
  | Internal huffman tree data.
  |
  */

crate::ix!();

pub struct HuffmanTreeHierarchyNode<T> {
    label:          T,
    count:          i32,
    left_ch_index:  i32,
    right_ch_index: i32,
}

impl<T> HuffmanTreeHierarchyNode<T> {
    fn new(l: T, count: i32) -> Self {
        Self {
            label:          l,
            count:          count,
            left_ch_index:  -1,
            right_ch_index: -1,
        }
    }
}

pub struct HuffmanTreeHierarchyNodeComparator;

impl HuffmanTreeHierarchyNodeComparator {

    #[inline] pub fn invoke<T>(
        &mut self, 
        node_a: &HuffmanTreeHierarchyNode<T>,
        node_b: &HuffmanTreeHierarchyNode<T>) -> bool 
    {
        todo!();
        /*
           return node_a.count > node_b.count;
           */
    }
}
