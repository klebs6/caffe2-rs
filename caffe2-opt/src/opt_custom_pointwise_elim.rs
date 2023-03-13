crate::ix!();


/**
  | -----------
  | @brief
  | 
  | This fuses Cast -> BatchOneHot -> Cast
  | into a single call.
  |
  */
#[inline] pub fn fuse_cast_batch_one_hot<T,U>(nn: *mut NNModule<T,U>)  {
    
    todo!();
    /*
        nom::nql::GraphMatcher gm;
      gm.initFromString(R"NQL(def nn {
          %cast = Cast(%input)
          %one_hot = BatchOneHot(%cast, %lengths, %values)
          %out = Cast(%one_hot)
      })NQL");
      CAFFE_ENFORCE(gm.getMatcher(), "Unable to parse NQL query.");

      for (const auto& match : gm.getMatches(nn->dataFlow)) {
        // This matches most of prod as of H2 2018
        auto first_cast = nn::getProducer(match["\%cast"]);
        auto second_cast = nn::getProducer(match["\%out"]);
        NOM_REQUIRE_OR_CONT(nn::get<Cast>(first_cast)->getTo() == 10);
        NOM_REQUIRE_OR_CONT(nn::get<Cast>(second_cast)->getTo() == 1);

        nn->replaceSubgraphWithOperator<CastedBatchOneHot>(
            match.subgraph,
            {match["\%input"], match["\%lengths"], match["\%values"]},
            {match["\%out"]});
      }
    */
}

register_opt_pass_from_func!{FuseCastBatchOneHot, fuseCastBatchOneHot}
