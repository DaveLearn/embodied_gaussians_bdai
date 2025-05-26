# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import abc
import typing
from dataclasses import dataclass

import torch
from typing import Any, Generic, NoReturn, TypeVar


@dataclass
class EnvironmentActions: ...


@dataclass
class EnvironmentObservations: ...

TEnvironmentActions = TypeVar("TEnvironmentActions", bound=EnvironmentActions)
TEnvironmentObservations = TypeVar("TEnvironmentObservations", bound=EnvironmentObservations)

class Environment(Generic[TEnvironmentActions, TEnvironmentObservations], abc.ABC):
    @abc.abstractmethod
    def observe(self) -> TEnvironmentObservations:
        """Get the current environment observations."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        ...

    @abc.abstractmethod
    def act(self, actions: TEnvironmentActions) -> None:
        """Apply actions to the environment."""
        ...

    @abc.abstractmethod
    def default_actions(self) -> TEnvironmentActions:
        """Get default actions for the environment."""
        ...

    @abc.abstractmethod
    def step(self) -> None:
        """Step the environment forward in time."""
        ...

    @abc.abstractmethod
    def dt(self) -> float:
        """Get the timestep size."""
        ...

    @abc.abstractmethod
    def time(self) -> float:
        """Get the current simulation time."""
        ...

    @abc.abstractmethod
    def num_envs(self) -> int:
        """Get the number of parallel environments."""
        ...


class Task:
    @abc.abstractmethod
    def done(self) -> torch.Tensor:
        """Return a boolean tensor indicating if each environment is done."""
        ...


class TaskEnvironment(Environment, Task): ...
