# deepspeed must be fully imported before transformers.modeling_utils: with
# deepspeed 0.19 + transformers 4.51, modeling_utils imports deepspeed at
# module init and deepspeed's hybrid_engine reaches back into
# transformers.models.opt at import -- a circular import that only resolves
# when deepspeed loads first.
try:
    import deepspeed  # noqa: F401
except ImportError:
    pass

