

def get_binary_function_list_from_op(op):
    def _fn0(x, y):
        return op(x, y)
    def _fn1(x, y):
        return op(op(op(op(x, y), y), x), y)
    return [
        _fn0,
        _fn1,
    ]

