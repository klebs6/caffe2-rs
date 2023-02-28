crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Cache.h]

pub trait HasDescriptor {
    type Descriptor;
}

pub trait HasHandle {
    type Handle;
}

pub trait HasHasher {
    type Hasher;
}

#[cfg(USE_VULKAN_API)]
pub use vulkan_api::*;

#[cfg(USE_VULKAN_API)]
mod vulkan_api {

    use super::*;

    pub mod vulkan_cache_configuration {
        pub const RESERVE: u32 = 64;
    }

    /**
      | A generic cache for immutable Vulkan
      | objects, when there will not be many
      | instances of those objects required at
      | runtime.
      |
      | The previous sentence puts two constraints on
      | proper use of this cache:
      |
      | 1) First, the objects should preferably be
      | immutable otherwise much care is required to
      | synchronize their usage.
      |
      | 2) Second, this cache is only intended for
      | objects that we will not have many instances of
      | during the entire execution of the program,
      | otherwise the cache must be _infrequently_
      | purged.
      |
      | Proper usage model for this cache is in direct
      | contrast with Vulkan object pools, which indeed
      | are required to be _frequently_ purged.  That
      | is an important distinction.
      |
      */
    pub struct VulkanCache<Factory: HasDescriptor + HasHandle + HasHasher> {
        cache:   FlatHashMap<Descriptor,Handle,Hasher>,
        factory: Factory,
    }

    impl<Factory> VulkanCache<Factory> 
    where Factory: HasDescriptor + HasHandle + HasHasher 
    {
        pub fn new(factory: Factory) -> Self {
        
            todo!();
            /*
          : factory_(move(factory)) 
            cache_.reserve(Configuration::kReserve);
            
            */
        }

        /**
          | Create or retrieve a resource.
          | 
          | This operation is a simple cache lookup
          | and returns the Handle corresponding
          | to the descriptor if the object is already
          | present in the cache. Otherwise,
          | 
          | Factory is used to create the object,
          | after which point the object is added
          | to the cache. Regardless, this function
          | returns with the object in the cache.
          |
          */
        pub fn retrieve(&mut self, descriptor: &Descriptor) -> Auto {
            
            todo!();
            /*
              auto iterator = cache_.find(descriptor);
              if C10_UNLIKELY(cache_.cend() == iterator) {
                iterator = cache_.insert({descriptor, factory_(descriptor)}).first;
              }

              return iterator->second.get();
            */
        }

        /**
          | Only call this function infrequently, if
          | ever.
          |
          | This cache is only intended for immutable
          | Vulkan objects of which a small finite
          | instances are required at runtime.
          |
          | A good place to call this function is between
          | model loads.
          */
        pub fn purge(&mut self)  {
            
            todo!();
            /*
               cache_.clear();
            */
        }
    }
}
