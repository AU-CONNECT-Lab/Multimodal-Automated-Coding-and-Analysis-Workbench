from macaw.interface.callbacks import (
    align_callbacks,
    config_callbacks,
    job_monitor_callbacks,
    pipeline_callbacks,
    review_callbacks,
)

MODULES = [
    pipeline_callbacks,
    align_callbacks,
    job_monitor_callbacks,
    config_callbacks,
    review_callbacks,
]
