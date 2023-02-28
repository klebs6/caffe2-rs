crate::ix!();

pub struct RunnableNet<'a> {

    netdef:      &'a NetDef,
    filler:      *const Filler,
    debug_info:  String,
}

impl<'a> RunnableNet<'a> {
    
    pub fn new(
        netdef: &NetDef,
        filler: *const Filler,
        info:   &String) -> Self {

        todo!();
        /*
            : netdef(netdef_), filler(filler_), debug_info(info_)
        */
    }
}

/**
  | An interface to supplier a pair of net
  | and its filler.
  | 
  | The net should be able to run once the
  | filler fills the workspace.
  | 
  | The supplier should take the ownership
  | of both net and filler.
  |
  */
pub trait NetSupplier {

    /// next() should be thread-safe
    fn next(&mut self) -> RunnableNet;
}

/**
  | A simple net supplier that always return
  | the same net and filler pair.
  |
  */
pub struct SingleNetSupplier {
    filler:  Box<Filler>,
    netdef:  NetDef,
}

impl SingleNetSupplier {
    
    pub fn new(filler: Box<Filler>, netdef: NetDef) -> Self {
    
        todo!();
        /*
            : filler_(std::move(filler)), netdef_(netdef)
        */
    }
}

impl NetSupplier for SingleNetSupplier {

    #[inline] fn next(&mut self) -> RunnableNet {
        
        todo!();
        /*
            return RunnableNet(netdef_, filler_.get());
        */
    }
}

/**
  | A simple net supplier that always return
  | the same net and filler pair.
  | 
  | The SingleLoadedNetSupplier contains
  | a shared ptr to a workspace with parameters
  | already loaded by net loader.
  |
  */
pub struct SingleLoadedNetSupplier {
    base: SingleNetSupplier,

    ws:   Arc<Workspace>,
}

impl NetSupplier for SingleLoadedNetSupplier {

    #[inline] fn next(&mut self) -> RunnableNet {
        self.base.next()
    }
}

impl SingleLoadedNetSupplier {

    pub fn new(
        filler: Box<Filler>,
        netdef: NetDef,
        ws:     Arc<Workspace>) -> Self {
    
        todo!();
        /*
            : SingleNetSupplier(std::move(filler), netdef), ws_(ws)
        */
    }
    
    #[inline] pub fn get_loaded_workspace(&mut self) -> Arc<Workspace> {
        
        todo!();
        /*
            return ws_;
        */
    }
}

///----------------------------------------
pub struct MutatingNetSupplier {

    lock:     parking_lot::RawMutex,
    core:     Box<dyn NetSupplier>,
    nets:     Vec<NetDef>,

    mutator:  fn(_u0: *mut NetDef) -> c_void,
}

impl NetSupplier for MutatingNetSupplier {

    #[inline] fn next(&mut self) -> RunnableNet {
        
        todo!();
        /*
            RunnableNet orig = core_->next();
        NetDef* new_net = nullptr;
        {
          std::lock_guard<std::mutex> guard(lock_);
          nets_.push_back(orig.netdef);
          new_net = &nets_.back();
        }
        mutator_(new_net);
        return RunnableNet(*new_net, orig.filler, orig.debug_info);
        */
    }
}

impl MutatingNetSupplier {

    pub fn new(
        core: Box<dyn NetSupplier>, 
        m:    fn(_u0: *mut NetDef) -> c_void) -> Self {
    
        todo!();
        /*
            : core_(std::move(core)), mutator_(m)
        */
    }
}
