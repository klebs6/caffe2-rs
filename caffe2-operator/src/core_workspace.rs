crate::ix!();

#[derive(Clone)]
pub struct StopOnSignal {

    handler: Arc<SignalHandler>,
}

impl StopOnSignal {
    
    #[inline] pub fn invoke(&mut self, iter: i32) -> bool {
        
        todo!();
        /*
            return handler_->CheckForSignals() != SignalHandler::Action::STOP;
        */
    }
}

impl Default for StopOnSignal {
    
    fn default() -> Self {
        todo!();
        /*
            : handler_(std::make_shared<SignalHandler>(
                SignalHandler::Action::STOP,
                SignalHandler::Action::STOP)
        */
    }
}

type ShouldContinue = fn(i32) -> bool;

pub struct WorkspaceBookkeeper {
    wsmutex:    parking_lot::RawMutex,
    workspaces: HashSet<Arc<Workspace>>,
}

impl Default for WorkspaceBookkeeper {
    fn default() -> Self {
        todo!();
    }
}

lazy_static!{
    static ref bookkeeper: Arc<WorkspaceBookkeeper> = Arc::new(WorkspaceBookkeeper::default());
}

/**
  | Workspace is a class that holds all the
  | related objects created during runtime:
  | (1) all blobs, and (2) all instantiated
  | networks.
  | 
  | It is the owner of all these objects and
  | deals with the scaffolding logistics.
  |
  */
pub struct Workspace {

    last_failed_op_net_position: Atomic<i32>, // {};

    blob_map:                     HashMap<String,Box<Blob>>,
    root_folder:                  String,
    shared:                       *const Workspace,
    forwarded_blobs:              HashMap<String,(*const Workspace,String)>,
    thread_pool:                  Box<ThreadPool>,
    thread_pool_creation_mutex:   parking_lot::RawMutex,
    bookkeeper:                   Arc<Bookkeeper>,
    net_map:                      HashMap<String,Box<NetBase>>,
}

unsafe impl Send for Workspace {}
unsafe impl Sync for Workspace {}

impl Default for Workspace {
    
    /**
      | Initializes an empty workspace.
      |
      */
    fn default() -> Self {
        todo!();
        /*
            : Workspace(".", nullptr
        */
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        todo!();
        /* 
        if (FLAGS_caffe2_print_blob_sizes_at_exit) {
          PrintBlobSizes();
        }
        // This is why we have a bookkeeper_ shared_ptr instead of a naked static! A
        // naked static makes us vulnerable to out-of-order static destructor bugs.
        std::lock_guard<std::mutex> guard(bookkeeper_->wsmutex);
        bookkeeper_->workspaces.erase(this);
       */
    }
}

impl Workspace {
    
    /**
      | Initializes an empty workspace with
      | the given root folder.
      | 
      | For any operators that are going to interface
      | with the file system, such as load operators,
      | they will write things under this root
      | folder given by the workspace.
      |
      */
    pub fn new_from_root_folder(root_folder: &String) -> Self {
        todo!();
        /*
            : Workspace(root_folder, nullptr)
        */
    }
    
    /**
      | Initializes a workspace with a shared
      | workspace.
      | 
      | When we access a Blob, we will first try
      | to access the blob that exists in the
      | local workspace, and if not, access
      | the blob that exists in the shared workspace.
      | 
      | The caller keeps the ownership of the
      | shared workspace and is responsible
      | for making sure that its lifetime is
      | longer than the created workspace.
      |
      */
    pub fn new_from_shared_workspace(shared: *const Workspace) -> Self {
        todo!();
        /*
            : Workspace(".", shared)
        */
    }
    
    /**
      | Initializes workspace with parent
      | workspace, blob name remapping (new
      | name -> parent blob name), no other blobs
      | are inherited from parent workspace
      |
      */
    pub fn new_from_parent_workspace_and_blob_remapping(
        shared: *const Workspace,
        forwarded_blobs: &HashMap<String,String>) -> Self 
    {
        todo!();
        /*
            : Workspace(".", nullptr) 

        CAFFE_ENFORCE(shared, "Parent workspace must be specified");
        for (const auto& forwarded : forwarded_blobs) {
          CAFFE_ENFORCE(
              shared->HasBlob(forwarded.second),
              "Invalid parent workspace blob: ",
              forwarded.second);
          forwarded_blobs_[forwarded.first] =
              std::make_pair(shared, forwarded.second);
        }
        */
    }
    
    /**
      | Initializes a workspace with a root
      | folder and a shared workspace.
      |
      */
    pub fn new_from_root_folder_and_shared_workspace(
        root_folder: &String,
        shared: *const Workspace) -> Self 
    {
        todo!();
        /*
            : root_folder_(root_folder), shared_(shared), bookkeeper_(bookkeeper()) 

        std::lock_guard<std::mutex> guard(bookkeeper_->wsmutex);
        bookkeeper_->workspaces.insert(this);
        */
    }

    /**
      | Converts previously mapped tensor
      | blobs to local blobs, copies values
      | from parent workspace blobs into new
      | local blobs. Ignores undefined blobs.
      |
      */
    #[inline] pub fn copy_forwarded_tensors<Context>(&mut self, blobs: &HashSet<String>)  {
        todo!();
        /*
            for (const auto& blob : blobs) {
          auto it = forwarded_blobs_.find(blob);
          if (it == forwarded_blobs_.end()) {
            continue;
          }
          const auto& ws_blob = it->second;
          const auto* parent_ws = ws_blob.first;
          auto* from_blob = parent_ws->GetBlob(ws_blob.second);
          CAFFE_ENFORCE(from_blob);
          CAFFE_ENFORCE(
              from_blob->template IsType<Tensor>(),
              "Expected blob with tensor value",
              ws_blob.second);
          forwarded_blobs_.erase(blob);
          auto* to_blob = CreateBlob(blob);
          CAFFE_ENFORCE(to_blob);
          const auto& from_tensor = from_blob->template Get<Tensor>();
          auto* to_tensor = BlobGetMutableTensor(to_blob, Context::GetDeviceType());
          to_tensor->CopyFrom(from_tensor);
        }
        */
    }
    
    /**
      | Return the root folder of the workspace.
      |
      */
    #[inline] pub fn root_folder(&mut self) -> &String {
        
        todo!();
        /*
            return root_folder_;
        */
    }
    
    /**
      | Checks if a blob with the given name is
      | present in the current workspace.
      |
      */
    #[inline] pub fn has_blob(&self, name: &String) -> bool {
        
        todo!();
        /*
            // First, check the local workspace,
        // Then, check the forwarding map, then the parent workspace
        if (blob_map_.count(name)) {
          return true;
        }

        auto it = forwarded_blobs_.find(name);
        if (it != forwarded_blobs_.end()) {
          const auto parent_ws = it->second.first;
          const auto& parent_name = it->second.second;
          return parent_ws->HasBlob(parent_name);
        }

        if (shared_) {
          return shared_->HasBlob(name);
        }

        return false;
        */
    }
    
    /**
      | Returns a list of names of the currently
      | instantiated networks.
      |
      */
    #[inline] pub fn nets(&self) -> Vec<String> {
        
        todo!();
        /*
            vector<string> names;
        for (auto& entry : net_map_) {
          names.push_back(entry.first);
        }
        return names;
        */
    }
    
    #[inline] pub fn print_blob_sizes(&mut self)  {
        
        todo!();
        /*
            vector<string> blobs = LocalBlobs();
      size_t cumtotal = 0;

      // First get total sizes and sort
      vector<std::pair<size_t, std::string>> blob_sizes;
      for (const auto& s : blobs) {
        Blob* b = this->GetBlob(s);
        TensorInfoCall shape_fun = GetTensorInfoFunction(b->meta().id());
        if (shape_fun) {
          size_t capacity;
          DeviceOption _device;
          auto shape = shape_fun(b->GetRaw(), &capacity, &_device);
          // NB: currently it overcounts capacity of shared storages
          // TODO: fix it after the storage sharing is merged
          cumtotal += capacity;
          blob_sizes.push_back(make_pair(capacity, s));
        }
      }
      std::sort(
          blob_sizes.begin(),
          blob_sizes.end(),
          [](const std::pair<size_t, std::string>& a,
             const std::pair<size_t, std::string>& b) {
            return b.first < a.first;
          });

      // Then print in descending order
      LOG(INFO) << "---- Workspace blobs: ---- ";
      LOG(INFO) << "name;current shape;capacity bytes;percentage";

      for (const auto& sb : blob_sizes) {
        Blob* b = this->GetBlob(sb.second);
        TensorInfoCall shape_fun = GetTensorInfoFunction(b->meta().id());
        CHECK(shape_fun != nullptr);
        size_t capacity;
        DeviceOption _device;

        auto shape = shape_fun(b->GetRaw(), &capacity, &_device);
        std::stringstream ss;
        ss << sb.second << ";";
        for (const auto d : shape) {
          ss << d << ",";
        }
        LOG(INFO) << ss.str() << ";" << sb.first << ";" << std::setprecision(3)
                  << (cumtotal > 0 ? 100.0 * double(sb.first) / cumtotal : 0.0)
                  << "%";
      }
      LOG(INFO) << "Total;;" << cumtotal << ";100%";
        */
    }
    
    /**
      | Return list of blobs owned by this
      | Workspace, not including blobs shared
      | from parent workspace.
      |
      */
    #[inline] pub fn local_blobs(&self) -> Vec<String> {
        
        todo!();
        /*
            vector<string> names;
      names.reserve(blob_map_.size());
      for (auto& entry : blob_map_) {
        names.push_back(entry.first);
      }
      return names;
        */
    }
    
    /**
      | Return a list of blob names.
      | 
      | This may be a bit slow since it will involve
      | creation of multiple temp variables.
      | 
      | For best performance, simply use HasBlob()
      | and GetBlob().
      |
      */
    #[inline] pub fn blobs(&self) -> Vec<String> {
        
        todo!();
        /*
            vector<string> names;
      names.reserve(blob_map_.size());
      for (auto& entry : blob_map_) {
        names.push_back(entry.first);
      }
      for (const auto& forwarded : forwarded_blobs_) {
        const auto* parent_ws = forwarded.second.first;
        const auto& parent_name = forwarded.second.second;
        if (parent_ws->HasBlob(parent_name)) {
          names.push_back(forwarded.first);
        }
      }
      if (shared_) {
        const auto& shared_blobs = shared_->Blobs();
        names.insert(names.end(), shared_blobs.begin(), shared_blobs.end());
      }
      return names;
        */
    }
    
    /**
      | Creates a blob of the given name.
      | 
      | The pointer to the blob is returned,
      | but the workspace keeps ownership of
      | the pointer.
      | 
      | If a blob of the given name already exists,
      | the creation is skipped and the existing
      | blob is returned.
      |
      */
    #[inline] pub fn create_blob(&mut self, name: &String) -> *mut Blob {
        
        todo!();
        /*
            if (HasBlob(name)) {
        VLOG(1) << "Blob " << name << " already exists. Skipping.";
      } else if (forwarded_blobs_.count(name)) {
        // possible if parent workspace deletes forwarded blob
        VLOG(1) << "Blob " << name << " is already forwarded from parent workspace "
                << "(blob " << forwarded_blobs_[name].second << "). Skipping.";
      } else {
        VLOG(1) << "Creating blob " << name;
        blob_map_[name] = unique_ptr<Blob>(new Blob());
      }
      return GetBlob(name);
        */
    }
    
    /**
      | Similar to CreateBlob(), but it creates
      | a blob in the local workspace even if
      | another blob with the same name already
      | exists in the parent workspace -- in
      | such case the new blob hides the blob
      | in parent workspace.
      | 
      | If a blob of the given name already exists
      | in the local workspace, the creation
      | is skipped and the existing blob is returned.
      |
      */
    #[inline] pub fn create_local_blob(&mut self, name: &String) -> *mut Blob {
        
        todo!();
        /*
            auto p = blob_map_.emplace(name, nullptr);
      if (!p.second) {
        VLOG(1) << "Blob " << name << " already exists. Skipping.";
      } else {
        VLOG(1) << "Creating blob " << name;
        p.first->second = std::make_unique<Blob>();
      }
      return p.first->second.get();
        */
    }
    
    /**
      | Renames a local workspace blob.
      | 
      | If blob is not found in the local blob
      | list or if the target name is already
      | present in local or any parent blob list
      | the function will throw.
      |
      */
    #[inline] pub fn rename_blob(
        &mut self, 
        old_name: &String,
        new_name: &String) -> *mut Blob 
    {
        todo!();
        /*
            // We allow renaming only local blobs for API clarity purpose
      auto it = blob_map_.find(old_name);
      CAFFE_ENFORCE(
          it != blob_map_.end(),
          "Blob ",
          old_name,
          " is not in the local blob list");

      // New blob can't be in any parent either, otherwise it will hide a parent
      // blob
      CAFFE_ENFORCE(
          !HasBlob(new_name), "Blob ", new_name, "is already in the workspace");

      // First delete the old record
      auto value = std::move(it->second);
      blob_map_.erase(it);

      auto* raw_ptr = value.get();
      blob_map_[new_name] = std::move(value);
      return raw_ptr;
        */
    }
    
    /**
      | Remove the blob of the given name. Return
      | true if removed and false if not exist.
      | 
      | Will NOT remove from the shared workspace.
      |
      */
    #[inline] pub fn remove_blob(&mut self, name: &String) -> bool {
        
        todo!();
        /*
            auto it = blob_map_.find(name);
      if (it != blob_map_.end()) {
        VLOG(1) << "Removing blob " << name << " from this workspace.";
        blob_map_.erase(it);
        return true;
      }

      // won't go into shared_ here
      VLOG(1) << "Blob " << name << " not exists. Skipping.";
      return false;
        */
    }
    
    /**
      | Gets the blob with the given name as a
      | const pointer.
      | 
      | If the blob does not exist, a nullptr
      | is returned.
      |
      */
    #[inline] pub fn get_blob(&self, name: &String) -> *const Blob {
        
        todo!();
        /*
            {
        auto it = blob_map_.find(name);
        if (it != blob_map_.end()) {
          return it->second.get();
        }
      }

      {
        auto it = forwarded_blobs_.find(name);
        if (it != forwarded_blobs_.end()) {
          const auto* parent_ws = it->second.first;
          const auto& parent_name = it->second.second;
          return parent_ws->GetBlob(parent_name);
        }
      }

      if (shared_) {
        if (auto blob = shared_->GetBlob(name)) {
          return blob;
        }
      }

      LOG(WARNING) << "Blob " << name << " not in the workspace.";
      // TODO(Yangqing): do we want to always print out the list of blobs here?
      // LOG(WARNING) << "Current blobs:";
      // for (const auto& entry : blob_map_) {
      //   LOG(WARNING) << entry.first;
      // }
      return nullptr;
        */
    }
    
    /**
      | Adds blob mappings from workspace to
      | the blobs from parent workspace.
      | 
      | Creates blobs under possibly new names
      | that redirect read/write operations
      | to the blobs in the parent workspace.
      | 
      | Arguments:
      | 
      | - parent - pointer to parent workspace
      | 
      | - forwarded_blobs - map from new blob
      | name to blob name in parent's
      | 
      | - workspace skip_defined_blob - if
      | set skips blobs with names that already
      | exist in the workspace, otherwise throws
      | exception
      |
      */
    #[inline] pub fn add_blob_mapping(
        &mut self, 
        parent:             *const Workspace,
        forwarded_blobs:    &HashMap<String,String>,
        skip_defined_blobs: Option<bool>)  
    {
        let skip_defined_blobs = skip_defined_blobs.unwrap_or(false);

        todo!();
        /*
            CAFFE_ENFORCE(parent, "Parent workspace must be specified");
      for (const auto& forwarded : forwarded_blobs) {
        CAFFE_ENFORCE(
            parent->HasBlob(forwarded.second),
            "Invalid parent workspace blob " + forwarded.second);
        if (forwarded_blobs_.count(forwarded.first)) {
          const auto& ws_blob = forwarded_blobs_[forwarded.first];
          CAFFE_ENFORCE_EQ(
              ws_blob.first, parent, "Redefinition of blob " + forwarded.first);
          CAFFE_ENFORCE_EQ(
              ws_blob.second,
              forwarded.second,
              "Redefinition of blob " + forwarded.first);
        } else {
          if (skip_defined_blobs && HasBlob(forwarded.first)) {
            continue;
          }
          CAFFE_ENFORCE(
              !HasBlob(forwarded.first), "Redefinition of blob " + forwarded.first);
          // Lazy blob resolution - store the parent workspace and
          // blob name, blob value might change in the parent workspace
          forwarded_blobs_[forwarded.first] =
              std::make_pair(parent, forwarded.second);
        }
      }
        */
    }
    
    #[inline] pub fn create_net_from_net_def_ref(
        &mut self, 
        net_def:   &NetDef, 
        overwrite: Option<bool>) -> *mut NetBase 
    {
        let overwrite = overwrite.unwrap_or(false);
        
        todo!();
        /*
            std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
      return CreateNet(tmp_net_def, overwrite);
        */
    }
    
    /**
      | Creates a network with the given NetDef,
      | and returns the pointer to the network.
      | 
      | If there is anything wrong during the
      | creation of the network, a nullptr is
      | returned.
      | 
      | The Workspace keeps ownership of the
      | pointer.
      | 
      | If there is already a net created in the
      | workspace with the given name, CreateNet
      | will overwrite it if overwrite=true
      | is specified.
      | 
      | Otherwise, an exception is thrown.
      |
      */
    #[inline] pub fn create_net(
        &mut self, 
        net_def:   &Arc<NetDef>,
        overwrite: Option<bool>) -> *mut NetBase 
    {
        let overwrite = overwrite.unwrap_or(false);
        
        todo!();
        /*
            CAFFE_ENFORCE(net_def->has_name(), "Net definition should have a name.");
      if (net_map_.count(net_def->name()) > 0) {
        if (!overwrite) {
          CAFFE_THROW(
              "I respectfully refuse to overwrite an existing net of the same "
              "name \"",
              net_def->name(),
              "\", unless you explicitly specify overwrite=true.");
        }
        VLOG(1) << "Deleting existing network of the same name.";
        // Note(Yangqing): Why do we explicitly erase it here? Some components of
        // the old network, such as an opened LevelDB, may prevent us from creating
        // a new network before the old one is deleted. Thus we will need to first
        // erase the old one before the new one can be constructed.
        net_map_.erase(net_def->name());
      }
      // Create a new net with its name.
      VLOG(1) << "Initializing network " << net_def->name();
      net_map_[net_def->name()] =
          unique_ptr<NetBase>(caffe2::CreateNet(net_def, this));
      if (net_map_[net_def->name()].get() == nullptr) {
        LOG(ERROR) << "Error when creating the network."
                   << "Maybe net type: [" << net_def->type() << "] does not exist";
        net_map_.erase(net_def->name());
        return nullptr;
      }
      return net_map_[net_def->name()].get();
        */
    }
    
    /**
      | Gets the pointer to a created net. The
      | workspace keeps ownership of the network.
      |
      */
    #[inline] pub fn get_net(&mut self, name: &String) -> *mut NetBase {
        
        todo!();
        /*
            auto it = net_map_.find(name);
      if (it != net_map_.end()) {
        return it->second.get();
      }

      return nullptr;
        */
    }
    
    /**
      | Deletes the instantiated network with
      | the given name.
      |
      */
    #[inline] pub fn delete_net(&mut self, name: &String)  {
        
        todo!();
        /*
            net_map_.erase(name);
        */
    }
    
    /**
      | Finds and runs the instantiated network
      | with the given name.
      | 
      | If the network does not exist or there
      | are errors running the network, the
      | function returns false.
      |
      */
    #[inline] pub fn run_net(&mut self, name: &String) -> bool {
        
        todo!();
        /*
            auto it = net_map_.find(name);
      if (it == net_map_.end()) {
        LOG(ERROR) << "Network " << name << " does not exist yet.";
        return false;
      }
      return it->second->Run();
        */
    }
    
    /**
      | RunOperatorOnce and RunNetOnce runs
      | an operator or net once.
      | 
      | The difference between RunNet and RunNetOnce
      | lies in the fact that RunNet allows you
      | to have a persistent net object, while
      | RunNetOnce creates a net and discards
      | it on the fly
      | 
      | - this may make things like database
      | read and random number generators repeat
      | the same thing over multiple calls.
      |
      */
    #[inline] pub fn run_operator_once(&mut self, 
        op_def: &OperatorDef) -> bool {
        
        todo!();
        /*
            std::unique_ptr<OperatorStorage> op(CreateOperator(op_def, this));
      if (op.get() == nullptr) {
        LOG(ERROR) << "Cannot create operator of type " << op_def.type();
        return false;
      }
      if (!op->Run()) {
        LOG(ERROR) << "Error when running operator " << op_def.type();
        return false;
      }
      // workaround for async cpu ops
      if (op->HasAsyncPart() && op->device_option().device_type() == PROTO_CPU) {
        op->Finish();
        return op->event().Query() == EventStatus::EVENT_SUCCESS;
      } else {
        return true;
      }
        */
    }
    
    #[inline] pub fn run_net_once(&mut self, 
        net_def: &NetDef) -> bool {
        
        todo!();
        /*
            std::unique_ptr<NetBase> net(caffe2::CreateNet(net_def, this));
      if (net == nullptr) {
        CAFFE_THROW(
            "Could not create net: " + net_def.name() + " of type " +
            net_def.type());
      }
      if (!net->Run()) {
        LOG(ERROR) << "Error when running network " << net_def.name();
        return false;
      }
      return true;
        */
    }
    
    /**
      | Runs a plan that has multiple nets and
      | execution steps.
      |
      */
    #[inline] pub fn run_plan(
        &mut self, 
        plan:            &PlanDef, 
        should_continue: Option<ShouldContinue>) -> bool 
    {
        let should_continue: ShouldContinue = todo!();//should_continue.unwrap_or(StopOnSignal::new());

        todo!();
        /*
            return RunPlanOnWorkspace(this, plan, shouldContinue);
        */
    }
    
    /**
      | Returns a CPU threadpool instance for
      | parallel execution of work.
      | 
      | The threadpool is created lazily; if
      | no operators use it, then no threadpool
      | will be created.
      |
      */
    #[inline] pub fn get_thread_pool(&mut self) -> *mut ThreadPool {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);
      if (!thread_pool_) {
        thread_pool_ = ThreadPool::defaultThreadPool();
      }
      return thread_pool_.get();
        */
    }
    
    #[inline] pub fn bookkeeper(&mut self) -> Arc<Bookkeeper> {
        
        todo!();
        /*
            static auto shared = std::make_shared<Workspace::Bookkeeper>();
      return shared;
        */
    }
    
    /**
      | Applies a function f on each workspace
      | that currently exists.
      | 
      | This function is thread safe and there
      | is no race condition between workspaces
      | being passed to f in this thread and destroyed
      | in another.
      |
      */
    #[inline] pub fn for_each<F>(&mut self, f: F) {
        todo!();
        /*
            auto bk = bookkeeper();
        std::lock_guard<std::mutex> guard(bk->wsmutex);
        for (Workspace* ws : bk->workspaces) {
          f(ws);
        }
        */
    }
}
