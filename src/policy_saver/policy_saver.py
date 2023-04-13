"""Policy Saver module"""

import logging

from tf_agents.agents.tf_agent import TFAgent
from tf_agents.policies.policy_saver import PolicySaver

from src import paths

logger = logging.getLogger(__name__)

POLICY_DIR = paths.BASE_DIR / "policy"


class TFPolicySaver:
    """PolicySaver wrapper class"""

    def __init__(self, name: str, agent: TFAgent):
        self._name = name
        self._agent = agent
        POLICY_DIR.mkdir(exist_ok=True)
        self._policy_dir = POLICY_DIR / name
        self._policy_dir.mkdir(exist_ok=True)
        self._policy_saver = PolicySaver(policy=self._agent.policy)

    def save(self) -> None:
        """Saves the policy in the directory"""
        self._policy_saver.save(export_dir=self._policy_dir)
