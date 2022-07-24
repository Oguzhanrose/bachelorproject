def assert_type(to_check, expected_type, allow_none:bool=False):
    """
    Check object against expected type

    :param to_check: Object for type check
    :param expected_type: Expected type of `to_check`
    :param allow_none: Weather or not None is an accepted type or not
    """

    if not isinstance(allow_none, bool):
        raise ValueError(f"Expected `allow_None` to by of type bool, but received type `{type(allow_none)}`")
    if (to_check is None) and (expected_type is None):
        raise TypeError(f"`None` is not a valid type. If you're trying to check if `type(to_check) == None` try set"
                        f" `expected_type=type(None)` instead.")

    is_ok = isinstance(to_check, expected_type)
    if allow_none:
        is_ok = (to_check is None) or is_ok

    if not is_ok:
        raise TypeError(f"Expected type `{expected_type}`, but received type `{type(to_check)}`")


def assert_types(to_check:list, expected_types:list, allow_nones:list=None):
    """
    Check list of values against expected types

    :param to_check: List of values for type check
    :param expected_types: Expected types of `to_check`
    :param allow_nones: list of booleans or 0/1
    """

    # Checks
    assert_type(to_check, list)
    assert_type(expected_types, list)
    assert_type(allow_nones, list, allow_none=True)
    if len(to_check) != len(expected_types):
        raise ValueError("length mismatch between `to_check_values` and `expected_types`")

    # If `allow_nones` is None all values are set to False.
    if allow_nones is None:
        allow_nones = [False for _ in range(len(to_check))]
    else:
        if len(allow_nones) != len(to_check):
            raise ValueError("length mismatch between `to_check_values` and `allow_nones`")
        for i, element in enumerate(allow_nones):
            if element in [0, 1]:
                allow_nones[i] = element == 1 # the `== 1` is just to allow for zeros as False and ones as True

    # check if all elements are of the correct type
    for i, value in enumerate(to_check):
        assert_type(value, expected_types[i], allow_nones[i])

