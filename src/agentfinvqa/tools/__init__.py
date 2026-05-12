"""Tools package: OCR reader, vision QA, and optional chart analysis helpers."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .ocr_reader_tool import OcrReaderTool
    from .vision_qa_tool import VisionQATool

__all__ = ["OcrReaderTool", "VisionQATool"]


def __getattr__(name: str) -> Any:
    if name == "OcrReaderTool":
        from .ocr_reader_tool import OcrReaderTool  # noqa: PLC0415

        return OcrReaderTool
    if name == "VisionQATool":
        from .vision_qa_tool import VisionQATool  # noqa: PLC0415

        return VisionQATool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
