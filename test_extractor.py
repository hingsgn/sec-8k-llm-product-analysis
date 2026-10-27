"""Offline regression checks for the boundaries that affect submission validity."""
import unittest
from sec_product_extractor import ground_product, recover_source_name, short_description


class ExtractionChecks(unittest.TestCase):
    def test_hallucinated_product_is_rejected(self):
        result, _ = ground_product('Imaginary Device', 'Acme announced quarterly earnings.')
        self.assertIsNone(result)

    def test_existing_product_mention_is_rejected(self):
        result, _ = ground_product('Alpha Server', 'Alpha Server revenue increased this quarter.')
        self.assertIsNone(result)

    def test_announcement_is_grounded_in_source(self):
        source = 'Acme introduced Alpha Server, a compact server for small offices.'
        result, _ = ground_product('Alpha Server', source)
        self.assertEqual(result['evidence'], source)
        self.assertEqual(source[result['name_start']:result['name_end']], 'Alpha Server')

    def test_trademark_variant_preserves_original_source_span(self):
        source = 'NVIDIA announced the NVIDIA ® GH200 Grace TM Hopper TM Superchip.'
        name = recover_source_name(source, 'NVIDIA GH200 Grace Hopper Superchip')
        self.assertEqual(name, 'NVIDIA ® GH200 Grace TM Hopper TM Superchip')
        self.assertIn(name, source)

    def test_strict_character_limit_includes_spaces(self):
        original = 'This product provides computing services for enterprise customers. ' * 5
        result = short_description(original)
        self.assertLess(len(result), 180)
        self.assertTrue(result.endswith('...'))


if __name__ == '__main__':
    unittest.main()
