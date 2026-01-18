import pytest
import os
import csv
import tempfile
import shutil
from pathlib import Path
from repos.stylegan2_ada_pytorch.generateGSCNFMod import append_manifest_row

class TestAppendManifestRow:
    
    def setup_method(self):
        """Setup temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.test_dir, 'test_manifest.csv')
    
    def teardown_method(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.test_dir)
    
    def read_csv_content(self, filepath):
        """Helper to read CSV content as list of lists."""
        with open(filepath, 'r', newline='') as f:
            return list(csv.reader(f))
    
    def test_create_new_manifest_with_header_and_row(self):
        """Test creating a new manifest file with header and first row."""
        header = ['gen_id', 'seed', 'img_path', 'w_npz_path']
        row = ['00000001', '1000', 'path/to/img.png', 'path/to/w.npz']
        
        append_manifest_row(self.manifest_path, header, row)
        
        # Verify file was created
        assert os.path.exists(self.manifest_path)
        
        # Verify content
        content = self.read_csv_content(self.manifest_path)
        assert len(content) == 2  # header + 1 row
        assert content[0] == header
        assert content[1] == row
    
    def test_append_to_existing_manifest_no_duplicate_header(self):
        """Test appending to existing manifest doesn't duplicate header."""
        header = ['gen_id', 'seed', 'img_path']
        row1 = ['00000001', '1000', 'img1.png']
        row2 = ['00000002', '1001', 'img2.png']
        
        # Create initial manifest
        append_manifest_row(self.manifest_path, header, row1)
        
        # Append second row
        append_manifest_row(self.manifest_path, header, row2)
        
        # Verify content
        content = self.read_csv_content(self.manifest_path)
        assert len(content) == 3  # header + 2 rows
        assert content[0] == header
        assert content[1] == row1
        assert content[2] == row2
    
    def test_multiple_appends_preserve_order(self):
        """Test multiple appends maintain correct order."""
        header = ['id', 'value']
        rows = [
            ['1', 'first'],
            ['2', 'second'], 
            ['3', 'third']
        ]
        
        # Append rows one by one
        for row in rows:
            append_manifest_row(self.manifest_path, header, row)
        
        # Verify all rows in correct order
        content = self.read_csv_content(self.manifest_path)
        assert len(content) == 4  # header + 3 rows
        assert content[0] == header
        for i, expected_row in enumerate(rows):
            assert content[i + 1] == expected_row
    
    def test_mixed_data_types_in_row(self):
        """Test handling of mixed data types (strings, numbers)."""
        header = ['gen_id', 'seed', 'z_dim', 'trunc']
        row = ['00000001', 1000, 512, 0.7]  # mixed types
        
        append_manifest_row(self.manifest_path, header, row)
        
        content = self.read_csv_content(self.manifest_path)
        # CSV writer converts everything to strings
        expected_row = ['00000001', '1000', '512', '0.7']
        assert content[1] == expected_row
    
    def test_empty_header_and_row(self):
        """Test handling of empty header and row."""
        header = []
        row = []
        
        append_manifest_row(self.manifest_path, header, row)
        
        content = self.read_csv_content(self.manifest_path)
        assert len(content) == 2
        assert content[0] == []
        assert content[1] == []
    
    def test_row_with_commas_and_quotes(self):
        """Test handling of rows containing commas and quotes."""
        header = ['id', 'description', 'path']
        row = ['001', 'test, with "quotes" and commas', 'path/with,comma.png']
        
        append_manifest_row(self.manifest_path, header, row)
        
        content = self.read_csv_content(self.manifest_path)
        assert content[1] == row  # CSV should handle escaping automatically
    
    def test_file_in_nonexistent_directory_raises_error(self):
        """Test that function raises appropriate error for non-existent directory."""
        nonexistent_path = os.path.join(self.test_dir, 'nonexistent', 'manifest.csv')
        header = ['id']
        row = ['1']
        
        with pytest.raises(FileNotFoundError):
            append_manifest_row(nonexistent_path, header, row)
    
    def test_realistic_stylegan_manifest_data(self):
        """Test with realistic StyleGAN manifest data structure."""
        header = ['gen_id','seed','img_path','w_npz_path','gene_class_idx','z_dim','w_dim','num_ws','trunc']
        
        rows = [
            ['00001000', '1000', 'data/generated/images/seed1000_latClass_0.png', 
             'data/generated/w/00001000.npz', '0', '512', '512', '16', '0.7'],
            ['00001001', '1001', 'data/generated/images/seed1001_latClass_0.png',
             'data/generated/w/00001001.npz', '0', '512', '512', '16', '0.7']
        ]
        
        for row in rows:
            append_manifest_row(self.manifest_path, header, row)
        
        content = self.read_csv_content(self.manifest_path)
        assert len(content) == 3  # header + 2 rows
        assert content[0] == header
        assert content[1] == rows[0]
        assert content[2] == rows[1]
    
    def test_header_row_length_mismatch(self):
        """Test behavior when header and row have different lengths."""
        header = ['id', 'name', 'value']
        row = ['1', 'test']  # Missing third value
        
        # Function should still work, CSV will handle the mismatch
        append_manifest_row(self.manifest_path, header, row)
        
        content = self.read_csv_content(self.manifest_path)
        assert content[0] == header
        assert content[1] == row
    
    def test_unicode_characters_in_data(self):
        """Test handling of unicode characters in data."""
        header = ['id', 'name', 'description']
        row = ['1', 'tëst_ñamé', 'déscriptiön with ümlauts']
        
        append_manifest_row(self.manifest_path, header, row)
        
        content = self.read_csv_content(self.manifest_path)
        assert content[1] == row
    
    def test_very_long_row_data(self):
        """Test handling of very long strings in row data."""
        header = ['id', 'long_path']
        very_long_path = 'a' * 1000  # Very long string
        row = ['1', very_long_path]
        
        append_manifest_row(self.manifest_path, header, row)
        
        content = self.read_csv_content(self.manifest_path)
        assert content[1] == row
        assert len(content[1][1]) == 1000