

def create_cluster(
    name: str, params: dict, 
    p0: float = 0.25, p: float = 0.2, r: int = 5, reeval_limit: int = 10):
    return __import__('clusters').__dict__[name](
        p0           = p0,
        p            = p,
        r            = r,
        reeval_limit = reeval_limit,
        **params
    )

    