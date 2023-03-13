crate::ix!();

#[test] fn lengths_to_segment_ids_op() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsToSegmentIds",
        ["lengths"],
        ["segment_ids"],
    )

    workspace.FeedBlob("lengths", np.array([1, 3, 0, 2]).astype(np.int32))
    print("lengths:\n", workspace.FetchBlob("lengths"))

    workspace.RunOperatorOnce(op)
    print("segment_ids: \n", workspace.FetchBlob("segment_ids"))

    lengths:
     [1 3 0 2]
    segment_ids:
     [0 1 1 1 3 3]

    */
}
