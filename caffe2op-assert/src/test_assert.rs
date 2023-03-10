crate::ix!();

#[test] fn assert_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Assert",
        ["A"],
        [],
        error_msg="Failed assertion from Assert operator"
    )

    workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.int32))
    print("A:", workspace.FetchBlob("A"))
    try:
        workspace.RunOperatorOnce(op)
    except RuntimeError:
        print("Assertion Failed!")
    else:
        print("Assertion Passed!")

    A:
    [[7 5 6]
     [1 2 4]
     [5 3 7]]
    Assertion Passed!

    */
}
