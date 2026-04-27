import importlib.util
import inspect
from pathlib import Path
from config import SKILLS_DIR

class Skill:
    def __init__(self, name, description, func, parameters=None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def execute(self, **kwargs):
        return self.func(**kwargs)

    def to_tool_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in self.parameters.items()},
                "required": list(self.parameters.keys())
            }
        }

class SkillRegistry:
    def __init__(self):
        self.skills = {}
        self._load_skills()

    def _load_skills(self):
        skills_dir = Path(SKILLS_DIR)
        for file in skills_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if hasattr(obj, "_is_skill"):
                    skill = Skill(obj._name, obj._description, obj, obj._parameters)
                    self.register(skill)

    def register(self, skill: Skill):
        self.skills[skill.name] = skill

    def get_tool_schemas(self):
        return [s.to_tool_schema() for s in self.skills.values()]

    def execute(self, name, **kwargs):
        if name not in self.skills:
            raise ValueError(f"Skill '{name}' not found")
        return self.skills[name].execute(**kwargs)

# Decorator to define a skill
def skill(name, description, parameters=None):
    def decorator(func):
        func._is_skill = True
        func._name = name
        func._description = description
        func._parameters = parameters or {}
        return func
    return decorator
