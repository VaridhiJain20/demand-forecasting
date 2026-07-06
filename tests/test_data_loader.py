import unittest

from src.data_loader import SUPPORTED_TABULAR_EXTENSIONS, load_any_file


class DataLoaderFormatTests(unittest.TestCase):
    def test_supported_extensions_are_limited_to_csv(self):
        self.assertEqual(SUPPORTED_TABULAR_EXTENSIONS, {"csv"})

    def test_csv_loading_still_works(self):
        csv_bytes = b"date,store,item,sales\n2024-01-01,1,1,10\n"
        df = load_any_file(csv_bytes, "sample.csv")
        self.assertEqual(list(df.columns), ["date", "store", "item", "sales"])
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
