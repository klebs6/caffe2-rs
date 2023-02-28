crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/VmapModeRegistrations.cpp]

/**
  | Note: [DispatchKey::VmapMode usage]
  |
  | Whenever we're inside a vmap, all Tensors
  | dispatch on this key. At the moment, this key
  | is used to disable random operations inside of
  | vmap.
  |
  | If you are looking for Batching Rules, those
  | are registered with DispatchKey::Batched
  | instead.
  |
  | Note: [Ambiguity of random operations inside
  | vmap]
  |
  | Random operations have an ambiguity where it
  | isn't clear if they should apply the same
  | randomness or apply different randomness.
  |
  | For example:
  |
  | >>> vmap(lambda t:
  | torch.rand(1))(torch.zeros(5)) Should the above
  | return the same random number 5 times, or
  | a different one?
  |
  | We haven't made a decision on that yet so we
  | are temporarily banning random operations
  | inside of vmap while we gather user feedback.
  */
pub fn unsupported_random_op<Args>(args: Args) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
                  "Please perform random operations outside of vmap as a workaround");
        */
}

pub fn unsupported_random_op_mut<Args>(args: Args) -> &mut Tensor {

    todo!();
        /*
            TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
                  "Please perform random operations outside of vmap as a workaround");
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_, VmapMode, m) {
      m.fallback(TorchCppFunction::makeFallthrough());
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, VmapMode, m) {
      // NB: I'd really like to register a special kernel like
      // CppFunction::makeNamedNotSupported() to avoid listing out the types of everything.
      // However, registering e.g. CppFunction::makeNamedNotSupported() as an implementation
      // only works for operators that support boxing.
    #define TENSOROPTIONS optional<ScalarType>, optional<Layout>, optional<Device>, optional<bool>

      // random operations (out-of-place)
      m.impl("bernoulli", unsupportedRandomOp<const Tensor&, optional<Generator>>);
      m.impl("bernoulli.out", unsupportedRandomOp_<const Tensor&, optional<Generator>, Tensor&>);
      m.impl("bernoulli.p", unsupportedRandomOp<const Tensor&, double, optional<Generator>>);
      m.impl("bernoulli_.Tensor", unsupportedRandomOp_<Tensor&, const Tensor&, optional<Generator>>);
      m.impl("bernoulli_.float", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);

      m.impl("cauchy_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);
      m.impl("exponential_", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);
      m.impl("geometric_", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);
      m.impl("log_normal_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);
      m.impl("multinomial", unsupportedRandomOp<const Tensor&, i64, bool, optional<Generator>>);
      m.impl("multinomial.out", unsupportedRandomOp_<const Tensor&, i64, bool, optional<Generator>, Tensor&>);

      m.impl("normal.Tensor_float", unsupportedRandomOp<const Tensor&, double, optional<Generator>>);
      m.impl("normal.Tensor_float_out", unsupportedRandomOp_<const Tensor&, double, optional<Generator>, Tensor&>);
      m.impl("normal.float_Tensor_out", unsupportedRandomOp_<double, const Tensor&, optional<Generator>, Tensor&>);
      m.impl("normal.float_Tensor", unsupportedRandomOp<double, const Tensor&, optional<Generator>>);
      m.impl("normal.Tensor_Tensor", unsupportedRandomOp<const Tensor&, const Tensor&, optional<Generator>>);
      m.impl("normal.Tensor_Tensor_out", unsupportedRandomOp_<const Tensor&, const Tensor&, optional<Generator>, Tensor&>);
      m.impl("normal.float_float", unsupportedRandomOp<double, double, IntArrayRef, optional<Generator>, TENSOROPTIONS>);
      m.impl("normal.float_float_out", unsupportedRandomOp_<double, double, IntArrayRef, optional<Generator>, Tensor&>);
      m.impl("normal_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);

      m.impl("poisson", unsupportedRandomOp<const Tensor&, optional<Generator>>);

      m.impl("random_.from", unsupportedRandomOp_<Tensor&, i64, optional<i64>, optional<Generator>>);
      m.impl("random_.to", unsupportedRandomOp_<Tensor&, i64, optional<Generator>>);
      m.impl("random_", unsupportedRandomOp_<Tensor&, optional<Generator>>);

      m.impl("rand_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, optional<MemoryFormat>>);
      m.impl("randn_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, optional<MemoryFormat>>);

      m.impl("randint_like", unsupportedRandomOp<const Tensor&, i64, TENSOROPTIONS, optional<MemoryFormat>>);
      m.impl("randint_like.low_dtype", unsupportedRandomOp<const Tensor&, i64, i64, TENSOROPTIONS, optional<MemoryFormat>>);

      m.impl("rand", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
      m.impl("rand.generator", unsupportedRandomOp<IntArrayRef, optional<Generator>, TENSOROPTIONS>);
      m.impl("rand.names", unsupportedRandomOp<IntArrayRef, optional<DimnameList>, TENSOROPTIONS>);
      m.impl("rand.generator_with_names", unsupportedRandomOp<IntArrayRef, optional<Generator>, optional<DimnameList>, TENSOROPTIONS>);
      m.impl("rand.out", unsupportedRandomOp_<IntArrayRef, Tensor&>);
      m.impl("rand.generator_out", unsupportedRandomOp_<IntArrayRef, optional<Generator>, Tensor&>);

      m.impl("randn", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
      m.impl("randn.generator", unsupportedRandomOp<IntArrayRef, optional<Generator>, TENSOROPTIONS>);
      m.impl("randn.names", unsupportedRandomOp<IntArrayRef, optional<DimnameList>, TENSOROPTIONS>);
      m.impl("randn.generator_with_names", unsupportedRandomOp<IntArrayRef, optional<Generator>, optional<DimnameList>, TENSOROPTIONS>);
      m.impl("randn.out", unsupportedRandomOp_<IntArrayRef, Tensor&>);
      m.impl("randn.generator_out", unsupportedRandomOp_<IntArrayRef, optional<Generator>, Tensor&>);

      m.impl("randperm", unsupportedRandomOp<i64, TENSOROPTIONS>);
      m.impl("randperm.generator", unsupportedRandomOp<i64, optional<Generator>, TENSOROPTIONS>);
      m.impl("randperm.out", unsupportedRandomOp_<i64, Tensor&>);
      m.impl("randperm.generator_out", unsupportedRandomOp_<i64, optional<Generator>, Tensor&>);

      m.impl("randint", unsupportedRandomOp<i64, IntArrayRef, TENSOROPTIONS>);
      m.impl("randint.generator", unsupportedRandomOp<i64, IntArrayRef, optional<Generator>, TENSOROPTIONS>);
      m.impl("randint.low", unsupportedRandomOp<i64, i64, IntArrayRef, TENSOROPTIONS>);
      m.impl("randint.low_generator", unsupportedRandomOp<i64, i64, IntArrayRef, optional<Generator>, TENSOROPTIONS>);
      m.impl("randint.out", unsupportedRandomOp_<i64, IntArrayRef, Tensor&>);
      m.impl("randint.generator_out", unsupportedRandomOp_<i64, IntArrayRef, optional<Generator>, Tensor&>);
      m.impl("randint.low_out", unsupportedRandomOp_<i64, i64, IntArrayRef, Tensor&>);
      m.impl("randint.low_generator_out", unsupportedRandomOp_<i64, i64, IntArrayRef, optional<Generator>, Tensor&>);

      m.impl("uniform_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);

    #undef TENSOROPTIONS
    }
    */
}
