"""Content pipeline — turns research into publishable output."""

from core.content.generator import ContentGenerator, GeneratedContent
from core.content.templates import REPORT_TEMPLATES, get_template

__all__ = ["ContentGenerator", "GeneratedContent", "REPORT_TEMPLATES", "get_template"]
