import contextlib
@contextlib.contextmanager
def autorelease_pool():
    yield
