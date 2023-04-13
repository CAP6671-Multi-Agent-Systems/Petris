"""Policy Saver module"""

import logging

from tf_agents.agents.tf_agent import TFAgent
from tf_agents.policies.policy_saver import PolicySaver

from src import paths

logger = logging.getLogger(__name__)

POLICY_DIR = paths.BASE_DIR / "policy"


def create_policy_saver(agent: TFAgent) -> PolicySaver:
    """
    Creates a new policy saver for the 'agent' as well as the 'policy' directory.

    Args:
        agent (TFAgent): Agent being trained

    Returns:
        PolicySaver: PolicySaver instance to use
    """
    
    POLICY_DIR.mkdir(exist_ok=True)
    
    return PolicySaver(policy=agent.policy)
