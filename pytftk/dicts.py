def dict_join(*dicts, overwrite_none=False, append=False):
    """Perform a non-commutative (left-to-right incremental) dict union.
    The result will have, for each common key, `{'k': 'v1'}, ..., {'k: 'vn'} -> {'k': 'vn'}`.

    Args:
        overwrite_none (bool, optional): If True, `None` values with on the same key will be replaced:
            `dict_union({'k': 'v'}, {'k': None}) == {'k': None}`, if False, they will be skipped.
            Defaults to False.

        append (bool, optional): If True, append values on matching keys instead of overwriting:
            `dict_join({'k': 'v1'}, {'k': 'v2'}) == {'k': ['v1', 'v2']}` and ignore the argument
            overwrite_none. The values of the resuslting dict are always flat lists. Defaults to False.

    Returns:
        dict: The union of the input dicts.
    """
    output_dict = {}
    for d in dicts:
        if append:
            for k in d:
                if k not in output_dict:
                    output_dict[k] = []
                output_dict[k] = output_dict[k] + (
                    d[k] if isinstance(d[k], list) else [d[k]]
                )
        else:
            output_dict = output_dict | {
                k: d[k] for k in d if d[k] is not None or overwrite_none
            }
    return output_dict
