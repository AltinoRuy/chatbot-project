def requires_context(expression, intent) -> bool:
    return expression is None and intent is not None