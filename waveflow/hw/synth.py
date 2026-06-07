def sim_only(fn):
    """Mark a method as simulation-only — skipped silently during synthesis.

    Use this on methods called inside ``run_proc`` that have no hardware
    meaning (logging, assertions, instrumentation).  Without this marker,
    any non-synthesizable call in ``run_proc`` raises ``SynthesisError``.

        @sim_only
        def log(self, **kwargs): ...
    """
    fn._is_sim_only = True
    return fn


def synthesizable(fn=None, *, synth_fn=None, stmt_class=None, impl_file=None):
    """Mark a method as synthesizable to HLS C++.

    May be used with or without parentheses::

        @synthesizable
        def method(self): ...

        @synthesizable(synth_fn=_gen_fn)
        def method(self): ...

        @synthesizable(impl_file="custom.cpp")
        def method(self): ...

    Parameters
    ----------
    synth_fn : callable or None
        Code-generation function ``(ctx, inputs, outputs) -> str``.  ``None``
        means stub behaviour — codegen emits a call to a user-written C++
        function in the impl file.
    stmt_class : type or None
        ``HwStmt`` subclass the extractor should instantiate for calls to this
        method.  ``None`` uses the default (``SynthCallStmt`` when *synth_fn*
        is set, ``FunctionStmt`` otherwise).
    impl_file : str or None
        Override the default impl-file location for ``FunctionStmt`` emission.
        ``None`` uses the default ``<component>_<function>_impl.cpp``.
    """
    def decorator(f):
        f._is_synthesizable = True
        f._synth_fn = synth_fn
        f._stmt_class = stmt_class
        f._impl_file = impl_file
        return f
    if fn is not None:
        return decorator(fn)
    return decorator
