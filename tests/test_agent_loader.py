"""Test agent_loader module."""

import pytest
from abotn_evaluator.agent_loader import load_class


class TestLoadClass:
    def test_valid_class(self):
        cls = load_class("agent_examples.point_goal_random:RandomPointGoalAgent")
        agent = cls()
        assert hasattr(agent, "reset")
        assert hasattr(agent, "predict")

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            load_class("no_colon_here")

    def test_nonexistent_module(self):
        with pytest.raises(ImportError):
            load_class("nonexistent.module:SomeClass")

    def test_nonexistent_class(self):
        with pytest.raises(AttributeError):
            load_class("agent_examples.point_goal_random:NonExistentClass")


class TestDualSystemDetection:
    def test_wrapper_agent(self):
        cls = load_class("agent_examples.point_goal_wrapper:PointGoalWrapperTemplate")
        agent = cls()
        assert hasattr(agent, "predict")
        assert hasattr(agent, "reset")
