crate::ix!();

pub trait BlobStatGetter {
    fn size_bytes(blob: &Blob) -> usize where Self: Sized;
}

pub struct BlobStatRegistrar<T, Getter> {
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Getter>,
}

impl<T,Getter> Default for BlobStatRegistrar<T, Getter> {
    
    fn default() -> Self {
        todo!();
        /*
            BlobStatRegistry::instance().doRegister(
              TypeMeta::Id<T>(), std::unique_ptr<Getter>(new Getter))
        */
    }
}

pub struct BlobStatRegistry {
    map: HashMap<TypeIdentifier, Box<dyn BlobStatGetter>>,
}

#[macro_export] macro_rules! register_blob_stat_getter {
    ($Type:ty, $BlobStatGetterClass:ty) => {
        /*
        static BlobStatRegistry::Registrar<$Type, $BlobStatGetterClass> 
            C10_ANONYMOUS_VARIABLE(BlobStatRegistry)
        */
    }
}

impl BlobStatRegistry {
    
    #[inline] pub fn get(&mut self, 
        id: TypeIdentifier) //TODO -> *const impl BlobStatGetter 
    {
        todo!();
        /*
            auto it = map_.find(id);
      if (it == map_.end()) {
        return nullptr;
      }
      return it->second.get();
        */
    }
    
    #[inline] pub fn instance(&mut self) -> &mut BlobStatRegistry {
        
        todo!();
        /*
            static BlobStatRegistry registry;
      return registry;
        */
    }
    
    #[inline] pub fn do_register(
        &mut self, 
        id: TypeIdentifier,
        v:  Box<dyn BlobStatGetter>)  
    {
        todo!();
        /*
            // don't use CAFFE_ENFORCE_EQ to avoid static initialization order fiasco.
      if (map_.count(id) > 0) {
        throw std::runtime_error("BlobStatRegistry: Type already registered.");
      }
      map_[id] = std::move(v);
        */
    }
    
    /**
     * Return size in bytes of the blob, if available for a blob of given type.
     * If not available, return 0.  */
    #[inline] pub fn size_bytes(&mut self, blob: &Blob) -> usize {
        
        todo!();
        /*
            auto* p = BlobStatRegistry::instance().get(blob.meta().id());
      return p ? p->sizeBytes(blob) : 0;
        */
    }
}
