crate::ix!();

#[test] fn and_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "And",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
    workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
     [[ True False False]
     [False  True False]
     [False False  True]]
    B:
     [[ True False  True]
     [False False False]
     [False False False]]
    C:
     [[ True False False]
     [False False False]
     [False False False]]

    */
}

#[test] fn or_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Or",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
    workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
    [[False  True  True]
     [False  True  True]
     [ True  True  True]]
    B:
    [[False  True False]
     [ True  True  True]
     [False  True False]]
    C:
    [[False  True  True]
     [ True  True  True]
     [ True  True  True]]

    */
}

#[test] fn xor_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Xor",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5))
    workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A:
    [[ True  True  True]
     [False False  True]
     [False  True False]]
    B:
    [[False False False]
     [ True  True  True]
     [False False False]]
    C:
    [[ True  True  True]
     [ True  True False]
     [False  True False]]
    */
}

