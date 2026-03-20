from scripts.evaluate_local import compute_composite_score


class TestCompositeScore:
    def test_formula(self):
        score = compute_composite_score(det_map=0.8, cls_map=0.5)
        assert abs(score - 0.71) < 1e-6

    def test_perfect_score(self):
        score = compute_composite_score(det_map=1.0, cls_map=1.0)
        assert abs(score - 1.0) < 1e-6

    def test_detection_only(self):
        score = compute_composite_score(det_map=1.0, cls_map=0.0)
        assert abs(score - 0.7) < 1e-6
