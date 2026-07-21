import logging
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich.panel import Panel


class RichYAMLHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.console = Console(width=120)

    def emit(self, record):
        if isinstance(record.msg, dict):
            yaml_str = yaml.dump(record.msg, sort_keys=False, allow_unicode=True)
            syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
            panel = Panel(
                syntax,
                title="[bold green]YAML Config[/bold green]",
                border_style="cyan",
            )
            self.console.print(panel)


class RichLogger:
    """Rich + YAML Logger"""

    def __init__(self, name="rich_logger", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not any(isinstance(h, RichHandler) for h in self.logger.handlers):
            rich_handler = RichHandler(
                console=Console(),
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
                omit_repeated_times=False,
            )
            rich_handler.setLevel(level)
            self.logger.addHandler(rich_handler)

        # YAML Handler
        if not any(isinstance(h, RichYAMLHandler) for h in self.logger.handlers):
            yaml_handler = RichYAMLHandler()
            yaml_handler.setLevel(logging.INFO)
            self.logger.addHandler(yaml_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def yaml(self, data: dict, level=logging.INFO):
        record = self.logger.makeRecord(
            self.logger.name, level, fn="", lno=0, msg=data, args=None, exc_info=None
        )
        for handler in self.logger.handlers:
            if isinstance(handler, RichYAMLHandler):
                handler.handle(record)


def get_logger(name="app", level=logging.DEBUG):
    return RichLogger(name, level)
