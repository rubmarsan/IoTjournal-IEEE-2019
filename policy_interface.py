import numpy as np
# from matplotlib import pyplot as plt


class PolicyIF( object ):
    """Interface for policy classes. Must implement state_to_action method"""

    def state_to_action(self, compound_state, x, y, z):
        raise NotImplementedError( "Should have implemented this" )