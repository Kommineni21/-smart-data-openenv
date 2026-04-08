import sys
import types

sys.modules['openenv.agent'] = types.ModuleType("agent")
sys.modules['openenv.agent.value_iteration'] = types.ModuleType("value_iteration")
sys.modules['openenv.util'] = types.ModuleType("util")
sys.modules['openenv.util.math'] = types.ModuleType("math")
sys.modules['openenv.util.math.boltzmann_softmax'] = types.ModuleType("boltzmann_softmax")
