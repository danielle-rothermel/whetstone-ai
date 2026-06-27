from __future__ import annotations

import logging

from dr_dspy.humaneval_direct_dbos import app
from dr_dspy.runtime import configure_multiprocessing

if __name__ == "__main__":
    configure_multiprocessing()
    logging.getLogger("dspy").setLevel(logging.WARNING)
    app()
