"""
Single slowapi Limiter instance shared across the entire application.

Import `limiter` wherever you need the `@limiter.limit(...)` decorator.
The app factory in app/main.py attaches this instance to `app.state.limiter`
and registers SlowAPIMiddleware, which is what slowapi requires to work.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
