/*!
  | An operation with N inputs and M outputs pops
  | the last N inputs off the stack and pushes its
  | M inputs onto the stack
  |
  | before: <other stack items> I0, I1, ... IN <- stack.back()
  |
  | after: <other stack items> O0, O1, ... OM
  |
  | operations are defined this way so that
  | ownership of inputs can be transferred to the
  | operation and it can incrementally drop
  | ownership of tensors when they become unneeded.
  |
  | For large operations, like 'run an entire
  | subgraph', this functionality is very important
  | for minimizing gpu memory usage return value is
  | the relative 'offset' to jump to for the next
  | operation:
  |
  |   pc += 1 + offset
  |
  | so a return value of 0 goes to the next
  | instruction
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/stack.h]

pub type Stack     = Vec<IValue>;
pub type Operation = fn(_0: *mut Stack) -> ();

/**
  | treat the last N elements of the stack
  | as a list, looking up element i
  |
  */
#[inline] pub fn peek_a(
    stack: &mut Stack,
    i:     usize,
    N:     usize) -> &mut IValue {

    todo!();
        /*
            return *(stack.end() - N + i);
        */
}

#[inline] pub fn peek_b<'a>(
        stack: *mut Stack,
        i:     usize,
        N:     usize) -> &'a mut IValue {
    
    todo!();
        /*
            return peek(*stack, i, N);
        */
}

#[inline] pub fn peek_c(
    stack: &Stack,
    i:     usize,
    N:     usize) -> &IValue {

    todo!();
        /*
            return *(stack.end() - N + i);
        */
}

#[inline] pub fn peek_d<'a>(
    stack: *const Stack,
    i:     usize,
    N:     usize) -> &'a IValue {

    todo!();
        /*
            return peek(*stack, i, N);
        */
}

/**
  | treat the last N elements of the stack as
  | a list, looking up the slice starting at index
  | i and having length len
  |
  */
#[inline] pub fn peek_slice(
        stack: &Stack,
        i:     usize,
        len:   usize,
        N:     usize) -> &[IValue] {
    
    todo!();
        /*
            return ArrayRef<IValue>(stack).slice(stack.size() - N + i, len);
        */
}


#[inline] pub fn last_a(
        stack: &Stack,
        N:     usize) -> &[IValue] {
    
    todo!();
        /*
            return peekSlice(stack, 0, N, N);
        */
}


#[inline] pub fn last_b<'a>(
        stack: *const Stack,
        N:     usize) -> &'a [IValue] {
    
    todo!();
        /*
            return last(*stack, N);
        */
}


#[inline] pub fn drop_a(
        stack: &mut Stack,
        n:     usize)  {
    
    todo!();
        /*
            stack.erase(stack.end() - n, stack.end());
        */
}


#[inline] pub fn drop_b(
        stack: *mut Stack,
        n:     usize)  {
    
    todo!();
        /*
            drop(*stack, n);
        */
}


#[inline] pub fn pop(stack: &mut Stack) -> IValue {
    
    todo!();
        /*
            auto r = move(stack.back());
      stack.pop_back();
      return r;
        */
}

#[inline] pub fn pop_n(
        stack: &mut Stack,
        n:     usize) -> Vec<IValue> {
    
    todo!();
        /*
            vector<IValue> result;
      result.reserve(n);
      for (usize i = 0; i < n; ++i) {
        result.push_back(move(peek(stack, i, n)));
      }
      drop(stack, n);
      return result;
        */
}

/**
  | variadic pop:
  |
  | i64 a; Tensor b;
  |
  | pop(stack, a, b);
  |
  | equivalent to:
  |
  | b = pop(stack).toTensor();
  |
  | a = pop(stack).toInt();
  |
  */
#[inline] pub fn pop_variadic<Types>(
    stack: &mut Stack,
    args:  &mut Types)  {

    todo!();
        /*
            usize i = 0;
      constexpr usize N = sizeof...(args);
      (void)initializer_list<int>{
          (args = move(peek(stack, i++, N)).template to<Types>(), 0)...};
      drop(stack, N);
        */
}

#[inline] pub fn push_one_a<Type>(
    stack: &mut Stack,
    arg:   Type)  {

    todo!();
        /*
            stack.emplace_back(forward<Type>(arg));
        */
}

#[inline] pub fn push_one_b(
    stack:   &mut Stack,
    options: TensorOptions)  {
    
    todo!();
        /*
            stack.emplace_back(typeMetaToScalarType(options.dtype()));
      stack.emplace_back(options.layout());
      stack.emplace_back(options.device());
      stack.emplace_back(options.pinned_memory());
        */
}

#[inline] pub fn push_a<Types>(
    stack: &mut Stack,
    args:  Types)  {

    todo!();
        /*
            (void)initializer_list<int>{(push_one(stack, forward<Types>(args)), 0)...};
        */
}

#[inline] pub fn push_b<Types>(
    stack: *mut Stack,
    args:  Types)  {

    todo!();
        /*
            return push(*stack, forward<Types>(args)...);
        */
}

#[inline] pub fn push_list_elements<T>(
        stack:    &mut Stack,
        elements: &List<T>)  {

    todo!();
        /*
            for (T elem : elements) {
        stack.push_back(move(elem));
      }
        */
}

/**
  | The packer here is carefully written not to
  | make any unnecessary copies.
  |
  | pack takes the return values of aten functions
  | pushes them onto the stack
  |
  */
#[inline] pub fn pack_a<T>(
    stack: &mut Stack,
    v:     T)  {

    todo!();
        /*
            stack.emplace_back(forward<T>(v));
        */
}

#[inline] pub fn pack_b<T>(
    stack: *mut Stack,
    v:     T)  {

    todo!();
        /*
            pack(*stack, forward<T>(v));
        */
}

pub struct TuplePacker<const remaining: usize,Args> {

}

impl<const remaining: usize,Args> TuplePacker<remaining,Args> {

    /**
      | NB: *Not* a universal reference.
      |
      */
    pub fn execute(
        stack: &mut Stack,
        t:     (Args))  {
        
        todo!();
        /*
            // NB: The move here does not "destroy" the entire tuple, that is
        // not what move does; only the particular tuple index
        // processed here gets stolen.
        pack(stack, get<sizeof...(Args) - remaining>(move(t)));
        TuplePacker<remaining - 1, Args...>::execute(stack, move(t));
        */
    }
}

lazy_static!{
    /*
    template <typename Args>
    struct TuplePacker<0, Args> {

      static void execute(Stack& stack, tuple<Args...>&& t){};
    };
    */
}

#[inline] pub fn pack_c<Args>(
    stack: &mut Stack,
    t:     (Args))  {

    todo!();
        /*
            TuplePacker<sizeof...(Args), Args...>::execute(stack, move(t));
        */
}
