.. toctree::

Core Concepts
==============

This document is a glossary for core concepts of *TorchRL* framework.

.. hint::

    The documentation for these classes is a work-in-progress. Be sure to
    checkout the source code for details.

.. _Learner:

Learner
--------------

:class:`Learner <torchrl.learners.base_learner.BaseLearner>` or Agent is an abstract
class which defines the learning agent in the given Environment_.

.. _Environment:

Environment
-------------

Environment is the system which provides feedback to the Learner_. Currently,
Open AI `gym.Env` environments are being used. The system is flexible enough
to extend to any other environment kind.


.. _Problem:

Problem
--------

Any task is defined by extending the abstract class
:class:`~torchrl.registry.problems.Problem`. A problem's entrypoint
is :meth:`~torchrl.registry.problems.Problem.run` which generates
the trajectory rollout and call's the Learner_'s
:meth:`~torchrl.learners.base_learner.BaseLearner.learn` method with
appropriate rollout information.


Hyper-Parameter Set
--------------------

A :class:`~torchrl.registry.hparams.HParams` set is a class of arbitrary
key-value pairs that contain the hyper-parameters for the problem. Keeping
these as first-class objects in the code base allow for easily reproducible
experiments.

Runner
-------

A :class:`~torchrl.episode_runner.MultiEpisodeRunner` takes in a
method which returns a constructed environment and creates multiple
subprocess copies for parallel trajectory rollouts via the
:meth:`~torchrl.episode_runner.MultiEpisodeRunner.collect` method. Each
Problem_ internally creates a :class:`~torchrl.episode_runner.MultiEpisodeRunner`
and executes the collection process.