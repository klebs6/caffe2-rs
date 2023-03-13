crate::ix!();

#[test] fn range_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Range",
        ["start", "stop", "step"],
        ["output"]
    )

    workspace.FeedBlob("start", np.array(4, dtype=np.int32))
    workspace.FeedBlob("stop", np.array(17, dtype=np.int32))
    workspace.FeedBlob("step", np.array(2, dtype=np.int32))
    print("start:", workspace.FetchBlob("start"))
    print("stop:", workspace.FetchBlob("stop"))
    print("step:", workspace.FetchBlob("step"))
    workspace.RunOperatorOnce(op)
    print("output:", workspace.FetchBlob("output"))

    start: 4
    stop: 17
    step: 2
    output: [ 4  6  8 10 12 14 16]
    */
}
