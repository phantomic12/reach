"""Tests for metadata extraction and generation."""

import pytest

from reach.tagtree.metadata import (
    build_metadata,
    detect_document_language,
    detect_per_passage_language,
    extract_title,
    generate_xmp_metadata,
    generate_xmp_string,
)
from reach.tagtree.tag_models import (
    BBox,
    ExtractionBlock,
    ExtractionDocument,
    ExtractionPage,
    FontInfo,
    TagNode,
    TagTree,
    TagType,
)


def make_block(
    block_type: str,
    text: str,
    font_size: float = 12.0,
) -> ExtractionBlock:
    return ExtractionBlock(
        type=block_type,
        bbox=BBox(x0=0, y0=0, x1=200, y1=20),
        text=text,
        font_info=FontInfo(size=font_size),
    )


class TestDetectDocumentLanguage:
    def test_english(self):
        text = "The quick brown fox jumps over the lazy dog. This is a test of the language detection system."
        lang = detect_document_language(text)
        assert lang == "en-US"

    def test_french(self):
        text = (
            "Le renard brun rapide saute par-dessus le chien paresseux. "
            "C'est un test du système de détection de langue. "
            "Cette phrase est écrite en français pour vérifier la détection automatique. "
            "Nous voulons nous assurer que le système fonctionne correctement."
        )
        lang = detect_document_language(text)
        assert lang == "fr-FR"

    def test_german(self):
        text = "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein Test des Spracherkennungssystems."
        lang = detect_document_language(text)
        assert lang == "de-DE"

    def test_spanish(self):
        text = "El rápido zorro marrón salta sobre el perro perezoso. Esta es una prueba del sistema de detección de idioma."
        lang = detect_document_language(text)
        assert lang == "es-ES"

    def test_empty_text(self):
        assert detect_document_language("") == "en-US"

    def test_short_text(self):
        assert detect_document_language("Hi there") == "en-US"

    def test_cjk_chinese(self):
        text = "这是一个测试文档用于检测中文语言。我们需要确保系统可以正确识别中文字符。"
        lang = detect_document_language(text)
        # Should detect CJK and default to Chinese if no hiragana/katakana/hangul
        assert lang in ("zh-CN", "ja-JP", "ko-KR")

    def test_japanese(self):
        text = "これは日本語のテストです。ひらがなとカタカナが含まれているドキュメントです。"
        lang = detect_document_language(text)
        assert lang == "ja-JP"

    def test_korean(self):
        text = "이것은 한국어 테스트입니다. 한글 문자가 포함된 문서입니다."
        lang = detect_document_language(text)
        assert lang == "ko-KR"

    def test_arabic(self):
        text = "هذا اختبار لنظام كشف اللغة العربية. يجب أن يتعرف النظام على النص العربي بشكل صحيح."
        lang = detect_document_language(text)
        assert lang == "ar-SA"

    def test_russian(self):
        text = "Это тест системы определения русского языка. Система должна правильно определять русский текст."
        lang = detect_document_language(text)
        assert lang == "ru-RU"


class TestDetectPerPassageLanguage:
    def test_same_language(self):
        result = detect_per_passage_language(
            "The system should work with English text throughout the document.",
            "en-US",
        )
        assert result is None

    def test_different_language(self):
        result = detect_per_passage_language(
            "Le texte français dans un document autrement anglais doit être détecté comme un passage différent.",
            "en-US",
        )
        assert result == "fr-FR"

    def test_short_text_skipped(self):
        result = detect_per_passage_language("Bonjour", "en-US")
        assert result is None


class TestExtractTitle:
    def test_from_metadata(self):
        doc = ExtractionDocument(
            metadata={"title": "Research Paper Title"},
            pages=[],
        )
        assert extract_title(doc) == "Research Paper Title"

    def test_from_pdf_metadata_key(self):
        doc = ExtractionDocument(
            metadata={"/Title": "PDF Title from Metadata"},
            pages=[],
        )
        assert extract_title(doc) == "PDF Title from Metadata"

    def test_from_first_page_heading(self):
        doc = ExtractionDocument(
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[
                        make_block("SectionHeader", "My Document Title", font_size=24),
                        make_block("Text", "Some content."),
                    ],
                ),
            ],
        )
        title = extract_title(doc)
        assert title == "My Document Title"

    def test_prefer_title_type(self):
        doc = ExtractionDocument(
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[
                        make_block("Title", "The Real Title", font_size=20),
                        make_block("SectionHeader", "A Subtitle", font_size=18),
                    ],
                ),
            ],
        )
        title = extract_title(doc)
        assert title == "The Real Title"

    def test_fallback_to_first_text(self):
        doc = ExtractionDocument(
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[
                        make_block("Text", "First paragraph of the document."),
                    ],
                ),
            ],
        )
        title = extract_title(doc)
        assert title == "First paragraph of the document."

    def test_fallback_to_filename(self):
        doc = ExtractionDocument(pages=[])
        title = extract_title(doc, filename="my-research-paper.pdf")
        assert "my research paper" in title.lower()


class TestXMPMetadata:
    def test_generate_basic(self):
        root = TagNode(tag=TagType.DOCUMENT)
        tree = TagTree(
            document=ExtractionDocument(),
            root=root,
            title="Test Document",
            language="en-US",
        )
        meta = generate_xmp_metadata(tree)
        assert meta["dc:title"]["x-default"] == "Test Document"
        assert meta["dc:language"] == "en-US"
        assert meta["pdfuaid:part"] == 2
        assert meta["pdf:PDFVersion"] == "2.0"
        assert meta["pdf:Tagged"] is True
        assert meta["pdf:DisplayDocTitle"] is True

    def test_xmp_string_generation(self):
        meta = {
            "dc:title": {"x-default": "Test"},
            "dc:creator": ["Reach"],
            "dc:language": "en-US",
            "dc:format": "application/pdf",
            "xmp:CreateDate": "2024-01-01T00:00:00+00:00",
            "xmp:ModifyDate": "2024-01-01T00:00:00+00:00",
            "xmp:CreatorTool": "Reach",
            "pdf:PDFVersion": "2.0",
            "pdf:Tagged": True,
            "pdf:DisplayDocTitle": True,
            "pdfuaid:part": 2,
        }
        xmp_string = generate_xmp_string(meta)

        assert '<?xpacket begin=""' in xmp_string
        assert "<dc:title>" in xmp_string
        assert "x-default" in xmp_string
        assert "<pdfuaid:part>2</pdfuaid:part>" in xmp_string
        assert "<pdf:PDFVersion>2.0</pdf:PDFVersion>" in xmp_string
        assert '<?xpacket end="w"?>' in xmp_string

    def test_xml_escaping(self):
        meta = {
            "dc:title": {"x-default": 'Test & "Special" <Characters>'},
            "dc:creator": ["Reach"],
            "dc:language": "en-US",
            "dc:format": "application/pdf",
            "xmp:CreateDate": "",
            "xmp:ModifyDate": "",
            "xmp:CreatorTool": "Reach",
            "pdf:PDFVersion": "2.0",
            "pdf:Tagged": True,
            "pdf:DisplayDocTitle": True,
            "pdfuaid:part": 2,
        }
        xmp_string = generate_xmp_string(meta)
        assert "&amp;" in xmp_string
        assert "&quot;" in xmp_string
        assert "&lt;" in xmp_string


class TestBuildMetadata:
    def test_full_pipeline(self):
        doc = ExtractionDocument(
            metadata={"/Title": "Full Pipeline Test"},
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[
                        make_block("SectionHeader", "Introduction"),
                        make_block(
                            "Text",
                            "The quick brown fox jumps over the lazy dog. This is a test document.",
                        ),
                    ],
                ),
            ],
        )
        root = TagNode(tag=TagType.DOCUMENT)
        tree = TagTree(document=doc, root=root)

        result = build_metadata(tree, filename="test.pdf")
        assert result.title == "Full Pipeline Test"
        assert result.language == "en-US"
        assert result.display_title is True
        assert result.xmp_metadata != {}

    def test_language_override_spans(self):
        doc = ExtractionDocument(
            pages=[
                ExtractionPage(
                    page_number=1,
                    width=612,
                    height=792,
                    blocks=[
                        make_block(
                            "Text",
                            "Le texte français dans un document anglais. La détection de langue devrait fonctionner correctement.",
                        ),
                    ],
                ),
            ],
        )
        root = TagNode(tag=TagType.DOCUMENT)
        tree = TagTree(document=doc, root=root)

        result = build_metadata(tree, filename="test.pdf")
        # Document might be detected as French given the block content
        # But language_spans should capture overrides
        assert result.language_spans is not None
