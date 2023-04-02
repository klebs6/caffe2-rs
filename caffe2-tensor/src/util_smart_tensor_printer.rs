crate::ix!();

/**
  | This is a wrapper around the TensorPrinter
  | that doesn't require the user to explicit
  | specify the type of the tensor while
  | calling the Print() method.
  | 
  | It also supports a convenience function
  | with a default constructed printer
  | as a static method.
  |
  */
#[derive(Default)]
pub struct SmartTensorPrinter<W: Write> {
    tensor_printer: TensorPrinter<W>,
}

impl<W: Write> SmartTensorPrinter<W> {
    
    pub fn new_from_tensorname(tensor_name: &String) -> Self {
        todo!();
        /*
            : tensorPrinter_(tensor_name)
        */
    }
    
    pub fn new_from_tensorname_and_filename(tensor_name: &String, file_name: &String) -> Self {
        todo!();
        /*
            : tensorPrinter_(tensor_name, file_name)
        */
    }
    
    pub fn new(
        tensor_name: &String,
        file_name:   &String,
        limit:       i32) -> Self 
    {
        todo!();
        /*
            : tensorPrinter_(tensor_name, file_name, limit)
        */
    }

    /**
      | Uses a default constructed
      | SmartTensorPrinter
      |
      */
    #[inline] pub fn print_tensor_meta(&mut self, tensor: &Tensor)  {
        
        todo!();
        /*
            DefaultTensorPrinter().PrintMeta(tensor);
        */
    }
    
    #[inline] pub fn print_meta(&mut self, tensor: &Tensor)  {
        
        todo!();
        /*
            tensorPrinter_.PrintMeta(tensor);
        */
    }
    
    #[inline] pub fn print(&mut self, tensor: &Tensor)  {
        
        todo!();
        /*
            ProxyPrinter printer;

      printer.tensor = &tensor;
      printer.tensorPrinter = &tensorPrinter_;
      printer.Print();
        */
    }
    
    /**
      | Returns a thread local default constructed
      | TensorPrinter
      |
      */
    #[inline] pub fn default_tensor_printer(&mut self) -> &mut SmartTensorPrinter<W> {
        
        todo!();
        /*
            // TODO(janusz): thread_local does not work under mac.
    #if defined(__APPLE__)
      CAFFE_THROW(
          "SmartTensorPrinter does not work on mac yet due to thread_local.");
    #else
      static thread_local SmartTensorPrinter printer;
      return printer;
    #endif
        */
    }
    
    /**
      | Uses a default constructed
      | SmartTensorPrinter
      |
      */
    #[inline] pub fn print_tensor(&mut self, tensor: &Tensor)  {
        
        todo!();
        /*
            DefaultTensorPrinter().Print(tensor);
        */
    }
}

/**
  | Since DispatchHelper doesn't support
  | passing arguments through the call()
  | method to DoRunWithType we have to create 
  | an object that will hold these arguments 
  | explicitly.
  |
  */
pub struct ProxyPrinter<W: Write> {
    tensor:         *const Tensor,
    tensor_printer: *mut TensorPrinter<W>,
}

impl<W: Write> ProxyPrinter<W> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            tensorPrinter->Print<T>(*tensor);
        return true;
        */
    }
    
    #[inline] pub fn print(&mut self)  {
        
        todo!();
        /*
            // Pulled in printable types from caffe2/core/types.cc
        // Unfortunately right now one has to add them by hand
        DispatchHelper<TensorTypes<
            float,
            int,
            std::string,
            bool,
            uint8_t,
            int8_t,
            uint16_t,
            int16_t,
            int64_t,
            double,
            char>>::call(this, tensor->dtype());
        */
    }
}
