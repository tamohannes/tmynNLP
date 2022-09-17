from typing import Dict, Any
from cores import Tracker
from aim.sdk.run import Run
import logging

logger = logging.getLogger(__name__)


@Tracker.register("aim")
class Aim(Tracker):
    def __init__(self, repo_path: str, experiment_name: str = None, system_tracking_interval: bool = True, log_system_params: bool = True) -> None:
        self.repo_path = repo_path
        self.experiment_name = experiment_name
        self.system_tracking_interval = system_tracking_interval
        self.log_system_params = log_system_params

        self._run = None
        self._run_hash = None

    def setup(self, args):
        if not self._run:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self.repo_path,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                )
            else:
                self._run = Run(
                    repo=self.repo_path,
                    experiment=self.experiment_name,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                )
                self._run_hash = self._run.hash

        # Log config parameters
        if args:
            try:
                for key in args:
                    if key == "description":
                        self._run.description = args[key]
                    else:
                        self._run.set(key, args[key], strict=False)
            except Exception as e:
                logger.warning(f'Aim could not log config parameters -> {e}')

    def experiment(self) -> Run:
        if not self._run:
            self.setup()
        return self._run

    def set_params(self, args: Dict[str, Any]):
        self.setup(args)

    def track(self, logs: Dict[str, Any] = None, context: Dict[str, str] = dict()):
        if logs:
            for log_name, log_value in logs.items():
                self._run.track(log_value, name=log_name, context=context)

    def __del__(self):
        if self._run is not None and self._run.active:
            self._run.close()
