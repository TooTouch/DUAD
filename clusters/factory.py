

def create_cluster(
    name: str, params: dict, 
    p0: float = 0.25, p: float = 0.2, r: int = 5):
    return __import__('clusters').__dict__[name](
        p0 = p0,
        p  = p,
        r  = r,
        **params
    )

    