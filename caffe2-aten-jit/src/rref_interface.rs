crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/rref_interface.h]

pub type TypePtr  = Arc<Type>;

pub type WorkerId = i16;

/**
  | This abstract class contains only user-facing
  | APIs, and will be shared between jit and
  | distributed to implement TorchScript support.
  |
  | RRef is made NOT copyable NOT movable to
  | prevent messing up reference counting.
  */
pub trait RRefInterface:
IntrusivePtrTarget
+ Default
+ Owner
+ OwnerName
+ IsOwner
+ ConfirmedByOwner
+ Ty {}

pub trait Owner {

    /**
      | returns the worker id of the owner
      |
      */
    fn owner(&self) -> WorkerId;
}

pub trait OwnerName {

    /**
      | returns the worker name of the owner
      |
      */
    fn owner_name(&self) -> String;
}

pub trait IsOwner {

    /**
      | Returns true if this is the ``OwnerRRef``
      |
      */
    fn is_owner(&self) -> bool;
}

pub trait ConfirmedByOwner {

    /**
      | Returns true if this is an ``OwnerRRef``
      | or if this ``UserRRef`` has been confirmed
      | by its owner.
      |
      */
    fn confirmed_by_owner(&self) -> bool;
}

pub trait Ty {

    fn ty(&self) -> TypePtr;
}
