# Try loading from local_config file first. If it doesn't exist, use default uploaded config.
try:
    from local_config import *
except:
    from uploadable_config import *