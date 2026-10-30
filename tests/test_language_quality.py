from legal_rag.language_quality import observed_answer_language


def test_german_answer_detected_for_english_question_example():
    german_answer = (
        "Eine natürliche Person qualifiziert sich als Verbraucher, wenn sie ein "
        "Rechtsgeschäft abschließt, das überwiegend weder ihrer gewerblichen noch "
        "ihrer selbständigen beruflichen Tätigkeit zugerechnet werden kann [S1]."
    )
    assert observed_answer_language(german_answer) == "de"


def test_english_answer_detected():
    assert observed_answer_language(
        "Under the BGB, a person attains legal adulthood at the age of 18 [S1]."
    ) == "en"


def test_short_or_multilingual_answer_is_unclassified():
    assert observed_answer_language("18 [S1]") is None
