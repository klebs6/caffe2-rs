crate::ix!();

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
