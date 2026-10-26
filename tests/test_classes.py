from fitodiag.classes import (CLASSES, SESAME_KEYWORDS, SHORT_LABELS, SOY_KEYWORDS,
                              match_keywords)


def test_fifteen_sorted_classes():
    assert len(CLASSES) == 15
    assert CLASSES == sorted(CLASSES)  # mismo orden que ImageFolder
    assert set(SHORT_LABELS) == set(CLASSES)


def test_soy_folder_names_from_dataset():
    cases = {
        "Caterpillar_Semilooper_Pest": "soja__oruga",
        "Frog_Leaf_Eye": "soja__ojo_de_rana",
        "Healthy": "soja__sana",
        "Mosaic": "soja__mosaico",
        "Rust": "soja__roya",
        "Spectoria_Brown_Spot": "soja__mancha_marron",
    }
    for folder, expected in cases.items():
        assert match_keywords([folder], SOY_KEYWORDS) == expected


def test_sesame_folder_names_from_dataset():
    cases = {
        "Healthy Leaf": "sesamo__sana",
        "Leaf Spot Disease": "sesamo__mancha_foliar",
        "Yellowing Leaf Syndrome": "sesamo__amarillamiento",
        "Insect Leaf Damage": "sesamo__insectos",
    }
    for folder, expected in cases.items():
        assert match_keywords(["Original Images", folder], SESAME_KEYWORDS) == expected
