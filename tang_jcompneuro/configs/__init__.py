"""
generic code to do type checking on my CNN parameters.

I can also implement all these using JSON schema. but that may involve too much engineering
and it's not very flexible from a practical perspective.
"""


# all checkers. either return True, or raise some Exception.
# NEVER return False or anything other than True.

def type_check_wrapper(x, type_checker, keys_to_check):
    assert isinstance(x, dict)
    assert x.keys() == keys_to_check, f'{x.keys()} NOT EQUAL to {keys_to_check}'
    for k, v in x.items():
        checker_this = type_checker[k]
        # notice that type objects are callable as well.
        if isinstance(checker_this, type):
            assert isinstance(v, checker_this)
        else:
            # https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
            assert callable(checker_this)
            # this should return True if everything is good.
            assert checker_this(v)
    return True
