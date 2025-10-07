# quick_check.py
import langextract as lx

print("langextract at:", lx.__file__)
lx.providers.load_plugins_once()              # <- important
print("REGISTRY ENTRIES:", lx.providers.registry.list_entries())
