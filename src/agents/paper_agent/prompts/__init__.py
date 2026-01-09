import os
from jinja2 import Environment, FileSystemLoader

_ENV = None

def get_template(template_name: str) -> str:
    global _ENV
    if _ENV is None:
        prompts_dir = os.path.dirname(os.path.abspath(__file__))
        _ENV = Environment(loader=FileSystemLoader(prompts_dir))
    
    template = _ENV.get_template(template_name)
    return template

def render_template(template_name: str, **kwargs) -> str:
    template = get_template(template_name)
    return template.render(**kwargs)
