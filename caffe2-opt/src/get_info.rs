crate::ix!();

#[inline] pub fn get_info_mut<'a,T,U>(
    infos: &'a mut HashMap<NodeRef<T,U>,GroupAnnotation>,
    node:  NodeRef<T,U>) -> &'a mut GroupAnnotation 
{
    todo!();
    /*
        auto it = infos.find(node);
      CAFFE_ENFORCE(it != infos.end(), "Node info not found for ", ShowNode(node));
      return it->second;
    */
}

#[inline] pub fn get_info<T,U>(
    infos: &HashMap<NodeRef<T,U>,GroupAnnotation>,
    node:  NodeRef<T,U>) -> &GroupAnnotation 
{
    todo!();
    /*
        auto it = infos.find(node);
      CAFFE_ENFORCE(
          it != infos.end(), "Const node info not found for ", ShowNode(node));
      return it->second;
    */
}

