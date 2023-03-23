crate::ix!();

#[inline] pub fn add_global_net_observer_creator(creator: NetObserverCreator)  {
    
    todo!();
    /*
        GetNetObserverCreators()->push_back(creator);
      VLOG(1) << "Have set a custom GlobalNetObserverCreator";
    */
}

#[inline] pub fn clear_global_net_observers()  {
    
    todo!();
    /*
        GetNetObserverCreators()->clear();
      VLOG(1) << "All net observers cleared";
    */
}

#[inline] pub fn get_net_observer_creators() -> *mut Vec<NetObserverCreator> {
    
    todo!();
    /*
        static std::vector<NetObserverCreator> creators;
      return &creators;
    */
}
