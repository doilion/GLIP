# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

import torch

from maskrcnn_benchmark.utils.imports import import_file


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file that performs
    custom setup work that may be necessary to their computing environment.
    """
    # Switch torch multiprocessing's tensor-sharing strategy from the default
    # ``file_descriptor`` to ``file_system``. With many DataLoader workers
    # (e.g. NUM_WORKERS * NPROC_PER_NODE = 6*8 = 48 on TCT_NGC), the FD-based
    # strategy hits per-process FD limits or transient shm-segment exhaustion
    # and crashes with errors like:
    #   "unable to open shared memory object </torch_…> in read-write mode"
    # The file_system strategy uses named files under /dev/shm instead of
    # passing FDs around — slightly higher overhead but no FD pressure.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        # Already set or not applicable in this context; safe to ignore.
        pass

    custom_module_path = os.environ.get("TORCH_DETECTRON_ENV_MODULE")
    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module_path):
    """Load custom environment setup from a Python source file and run the setup
    function.
    """
    module = import_file("maskrcnn_benchmark.utils.env.custom_module", custom_module_path)
    assert hasattr(module, "setup_environment") and callable(
        module.setup_environment
    ), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(
        custom_module_path
    )
    module.setup_environment()


# Force environment setup when this module is imported
setup_environment()
