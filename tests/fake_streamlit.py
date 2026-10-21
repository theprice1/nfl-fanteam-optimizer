"""Minimal stand-in for streamlit so app.py can be executed top to bottom in tests (logic check only)."""
import sys, types
import numpy as np, pandas as pd

class Ctx:
    def __init__(self, st): self._st = st
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(self._st, name)

class State(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v

class Cfg:
    def __getattr__(self, name): return lambda *a, **k: {"cfg": name, "args": a}

class Prog:
    def progress(self, *a, **k): pass
    def empty(self): pass

class Fake:
    def __init__(self, press=()):
        self.session_state = State(); self.press = set(press); self.column_config = Cfg(); self.sidebar = Ctx(self)
        self.calls = []
    def cache_data(self, *a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
    def stop(self): raise SystemExit
    def tabs(self, names): return [Ctx(self) for _ in names]
    def columns(self, spec, **k): return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def expander(self, *a, **k): return Ctx(self)
    def container(self): return Ctx(self)
    def progress(self, *a, **k): return Prog()
    def _w(self, key, default):
        return self.session_state[key] if key is not None and key in self.session_state else default
    def number_input(self, label, mn=None, mx=None, value=None, step=None, key=None, **k): return self._w(key, value)
    def slider(self, label, mn=None, mx=None, value=None, step=None, key=None, **k):
        return self._w(key, value if value is not None else mn)
    def select_slider(self, label, options, value=None, key=None, **k): return self._w(key, value)
    def toggle(self, label, value=False, key=None, **k): return self._w(key, value)
    def text_input(self, label, value="", key=None, **k): return self._w(key, value)
    def text_area(self, label, key=None, **k): return self._w(key, "")
    def radio(self, label, options, key=None, **k): return self._w(key, options[0])
    def multiselect(self, label, options, default=None, key=None, **k): return self._w(key, list(default or []))
    def file_uploader(self, *a, **k): return None
    def button(self, label, key=None, on_click=None, **k):
        if key in self.press or label in self.press:
            if on_click: on_click()
            return True
        return False
    def data_editor(self, df, **k): return df
    def download_button(self, *a, **k): self.calls.append(("download", a[1] if len(a) > 1 else None))
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a[:1]))

def install(press=()):
    f = Fake(press)
    mod = types.ModuleType("streamlit")
    mod.__getattr__ = lambda name: getattr(f, name)   # proxy every attribute to the fake
    sys.modules["streamlit"] = mod
    return f
