crate::ix!();

/**
  | These #defines are useful when writing passes
  | as the collapse
  |
  | if (!cond) {
  |   continue; // or break; or return;
  | }
  |
  | into a single line without negation
  */
#[macro_export] macro_rules! nom_require_or_ {
    ($_cond:ident, $_expr:ident) => {
        todo!();
        /*
        
          if (!(_cond)) {                     
            _expr;                            
          }
        */
    }
}

#[macro_export] macro_rules! nom_require_or_cont {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, continue)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_break {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, break)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret_null {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return nullptr)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret_false {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return false)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret_true {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return true)
        */
    }
}

#[macro_export] macro_rules! nom_require_or_ret {
    ($_cond:ident) => {
        todo!();
        /*
                NOM_REQUIRE_OR_(_cond, return )
        */
    }
}

/**
  | Implements accessors for a generic type T. If
  | the type is not specified (i.e., void template
  | type) then the partial specification gives an
  | empty type.
  */
pub struct StorageType<T> {
    data: T,
}

impl<T> StorageType<T> {
    
    pub fn new(data: T) -> Self {
    
        todo!();
        /*
            : Data(std::move(data))
        */
    }
    
    #[inline] pub fn data(&self) -> &T {
        
        todo!();
        /*
            return Data;
        */
    }
    
    #[inline] pub fn mutable_data(&mut self) -> *mut T {
        
        todo!();
        /*
            return &Data;
        */
    }
    
    #[inline] pub fn reset_data(&mut self, data: T)  {
        
        todo!();
        /*
            Data = std::move(data);
        */
    }
}

/**
  | -----------
  | @brief
  | 
  | This class enables a listener pattern.
  | 
  | It is to be used with a "curious recursive
  | pattern" i.e. Derived : public Notifier<Derived>
  | {}
  |
  */
pub struct Notifier<T> {
    dtor_callbacks:  LinkedList<NotifierCallback<T>>,
    notif_callbacks: LinkedList<NotifierCallback<T>>,
}

pub type NotifierCallback<T> = fn(_u0: *mut T) -> c_void;

impl<T> Notifier<T> {
    
    #[inline] pub fn register_destructor_callback(&mut self, fn_: NotifierCallback<T>) -> *mut NotifierCallback<T> {
        
        todo!();
        /*
            dtorCallbacks_.emplace_back(fn);
        return &dtorCallbacks_.back();
        */
    }
    
    #[inline] pub fn register_notification_callback(&mut self, fn_: NotifierCallback<T>) -> *mut NotifierCallback<T> {
        
        todo!();
        /*
            notifCallbacks_.emplace_back(fn);
        return &notifCallbacks_.back();
        */
    }
    
    #[inline] pub fn delete_callback(&mut self, callback_list: &mut LinkedList<NotifierCallback<T>>, to_delete: *mut NotifierCallback<T>)  {
        
        todo!();
        /*
            for (auto i = callbackList.begin(); i != callbackList.end(); ++i) {
          if (&*i == toDelete) {
            callbackList.erase(i);
            break;
          }
        }
        */
    }
    
    #[inline] pub fn delete_destructor_callback(&mut self, c: *mut NotifierCallback<T>)  {
        
        todo!();
        /*
            deleteCallback(dtorCallbacks_, c);
        */
    }
    
    #[inline] pub fn delete_notification_callback(&mut self, c: *mut NotifierCallback<T>)  {
        
        todo!();
        /*
            deleteCallback(notifCallbacks_, c);
        */
    }

    /**
      | \brief Notifies all listeners
      | (`registerNotificationCallback` users) of
      | an update.
      |
      | Assumes the information of the update is
      | encoded in the state of the derived class,
      | thus only passing a pointer of type T* to
      | the callback.
      */
    #[inline] pub fn notify(&mut self)  {
        
        todo!();
        /*
            for (auto callback : notifCallbacks_) {
          callback(reinterpret_cast<T*>(this));
        }
        */
    }
}

impl<T> Drop for Notifier<T> {
    fn drop(&mut self) {
        todo!();
        /* 
        for (auto callback : dtorCallbacks_) {
          callback(reinterpret_cast<T*>(this));
        }
       */
    }
}
