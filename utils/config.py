def updateConfig(obj, ref):
    if isinstance(ref, dict):
        for member, value in ref.items():
            setattr(obj, member, value)