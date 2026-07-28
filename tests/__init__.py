import inspect

def get_current_test_output() -> str:
    """Get the current test class and method name."""
    frame = inspect.currentframe().f_back
    class_name = frame.f_locals.get('self').__class__.__name__
    method_name = frame.f_code.co_name
    return f'./.tests-output/{class_name}_{method_name}'
