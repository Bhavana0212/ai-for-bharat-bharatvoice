<<<<<<< HEAD
"""
File compression utilities for storage optimization.
"""

import gzip
import logging
import lzma
import zlib
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"


class FileCompression:
    """File compression and decompression utilities."""
    
    def __init__(self, default_type: CompressionType = CompressionType.GZIP):
        """
        Initialize file compression.
        
        Args:
            default_type: Default compression type
        """
        self.default_type = default_type
    
    def compress_data(
        self, 
        data: bytes, 
        compression_type: Optional[CompressionType] = None
    ) -> bytes:
        """
        Compress data.
        
        Args:
            data: Data to compress
            compression_type: Compression type to use
            
        Returns:
            Compressed data
        """
        if compression_type is None:
            compression_type = self.default_type
        
        try:
            if compression_type == CompressionType.NONE:
                return data
            elif compression_type == CompressionType.GZIP:
                return gzip.compress(data, compresslevel=6)
            elif compression_type == CompressionType.LZMA:
                return lzma.compress(data, preset=6)
            elif compression_type == CompressionType.ZLIB:
                return zlib.compress(data, level=6)
            else:
                logger.warning(f"Unknown compression type: {compression_type}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to compress data with {compression_type}: {e}")
            return data
    
    def decompress_data(
        self, 
        compressed_data: bytes, 
        compression_type: CompressionType
    ) -> bytes:
        """
        Decompress data.
        
        Args:
            compressed_data: Compressed data
            compression_type: Compression type used
            
        Returns:
            Decompressed data
        """
        try:
            if compression_type == CompressionType.NONE:
                return compressed_data
            elif compression_type == CompressionType.GZIP:
                return gzip.decompress(compressed_data)
            elif compression_type == CompressionType.LZMA:
                return lzma.decompress(compressed_data)
            elif compression_type == CompressionType.ZLIB:
                return zlib.decompress(compressed_data)
            else:
                logger.warning(f"Unknown compression type: {compression_type}")
                return compressed_data
                
        except Exception as e:
            logger.error(f"Failed to decompress data with {compression_type}: {e}")
            raise
    
    def compress_file(
        self, 
        input_path: str, 
        output_path: str,
        compression_type: Optional[CompressionType] = None
    ) -> bool:
        """
        Compress a file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output compressed file
            compression_type: Compression type to use
            
        Returns:
            True if successful
        """
        if compression_type is None:
            compression_type = self.default_type
        
        try:
            with open(input_path, 'rb') as infile:
                data = infile.read()
            
            compressed_data = self.compress_data(data, compression_type)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(compressed_data)
            
            logger.debug(f"File compressed: {input_path} -> {output_path} ({compression_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compress file {input_path}: {e}")
            return False
    
    def decompress_file(
        self, 
        input_path: str, 
        output_path: str,
        compression_type: CompressionType
    ) -> bool:
        """
        Decompress a file.
        
        Args:
            input_path: Path to compressed file
            output_path: Path to output decompressed file
            compression_type: Compression type used
            
        Returns:
            True if successful
        """
        try:
            with open(input_path, 'rb') as infile:
                compressed_data = infile.read()
            
            data = self.decompress_data(compressed_data, compression_type)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(data)
            
            logger.debug(f"File decompressed: {input_path} -> {output_path} ({compression_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decompress file {input_path}: {e}")
            return False
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_size: Original data size
            compressed_size: Compressed data size
            
        Returns:
            Compression ratio (0.0 to 1.0, lower is better)
        """
        if original_size == 0:
            return 1.0
        
        return compressed_size / original_size
    
    def choose_best_compression(self, data: bytes) -> CompressionType:
        """
        Choose the best compression type for given data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Best compression type
        """
        if len(data) < 1024:  # Small files, don't compress
            return CompressionType.NONE
        
        # Test different compression types
        results = {}
        
        for comp_type in [CompressionType.GZIP, CompressionType.LZMA, CompressionType.ZLIB]:
            try:
                compressed = self.compress_data(data, comp_type)
                ratio = self.get_compression_ratio(len(data), len(compressed))
                results[comp_type] = ratio
            except Exception as e:
                logger.warning(f"Failed to test compression {comp_type}: {e}")
                results[comp_type] = 1.0
        
        # Choose the best ratio (lowest)
        best_type = min(results.keys(), key=lambda k: results[k])
        
        # If compression doesn't help much, don't compress
        if results[best_type] > 0.9:
            return CompressionType.NONE
        
        return best_type


# Global compression instance
_file_compression: Optional[FileCompression] = None


def get_file_compression() -> FileCompression:
    """Get global file compression instance."""
    global _file_compression
    
    if _file_compression is None:
        _file_compression = FileCompression()
    
=======
"""
File compression utilities for storage optimization.
"""

import gzip
import logging
import lzma
import zlib
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"


class FileCompression:
    """File compression and decompression utilities."""
    
    def __init__(self, default_type: CompressionType = CompressionType.GZIP):
        """
        Initialize file compression.
        
        Args:
            default_type: Default compression type
        """
        self.default_type = default_type
    
    def compress_data(
        self, 
        data: bytes, 
        compression_type: Optional[CompressionType] = None
    ) -> bytes:
        """
        Compress data.
        
        Args:
            data: Data to compress
            compression_type: Compression type to use
            
        Returns:
            Compressed data
        """
        if compression_type is None:
            compression_type = self.default_type
        
        try:
            if compression_type == CompressionType.NONE:
                return data
            elif compression_type == CompressionType.GZIP:
                return gzip.compress(data, compresslevel=6)
            elif compression_type == CompressionType.LZMA:
                return lzma.compress(data, preset=6)
            elif compression_type == CompressionType.ZLIB:
                return zlib.compress(data, level=6)
            else:
                logger.warning(f"Unknown compression type: {compression_type}")
                return data
                
        except Exception as e:
            logger.error(f"Failed to compress data with {compression_type}: {e}")
            return data
    
    def decompress_data(
        self, 
        compressed_data: bytes, 
        compression_type: CompressionType
    ) -> bytes:
        """
        Decompress data.
        
        Args:
            compressed_data: Compressed data
            compression_type: Compression type used
            
        Returns:
            Decompressed data
        """
        try:
            if compression_type == CompressionType.NONE:
                return compressed_data
            elif compression_type == CompressionType.GZIP:
                return gzip.decompress(compressed_data)
            elif compression_type == CompressionType.LZMA:
                return lzma.decompress(compressed_data)
            elif compression_type == CompressionType.ZLIB:
                return zlib.decompress(compressed_data)
            else:
                logger.warning(f"Unknown compression type: {compression_type}")
                return compressed_data
                
        except Exception as e:
            logger.error(f"Failed to decompress data with {compression_type}: {e}")
            raise
    
    def compress_file(
        self, 
        input_path: str, 
        output_path: str,
        compression_type: Optional[CompressionType] = None
    ) -> bool:
        """
        Compress a file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output compressed file
            compression_type: Compression type to use
            
        Returns:
            True if successful
        """
        if compression_type is None:
            compression_type = self.default_type
        
        try:
            with open(input_path, 'rb') as infile:
                data = infile.read()
            
            compressed_data = self.compress_data(data, compression_type)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(compressed_data)
            
            logger.debug(f"File compressed: {input_path} -> {output_path} ({compression_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compress file {input_path}: {e}")
            return False
    
    def decompress_file(
        self, 
        input_path: str, 
        output_path: str,
        compression_type: CompressionType
    ) -> bool:
        """
        Decompress a file.
        
        Args:
            input_path: Path to compressed file
            output_path: Path to output decompressed file
            compression_type: Compression type used
            
        Returns:
            True if successful
        """
        try:
            with open(input_path, 'rb') as infile:
                compressed_data = infile.read()
            
            data = self.decompress_data(compressed_data, compression_type)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(data)
            
            logger.debug(f"File decompressed: {input_path} -> {output_path} ({compression_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decompress file {input_path}: {e}")
            return False
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_size: Original data size
            compressed_size: Compressed data size
            
        Returns:
            Compression ratio (0.0 to 1.0, lower is better)
        """
        if original_size == 0:
            return 1.0
        
        return compressed_size / original_size
    
    def choose_best_compression(self, data: bytes) -> CompressionType:
        """
        Choose the best compression type for given data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Best compression type
        """
        if len(data) < 1024:  # Small files, don't compress
            return CompressionType.NONE
        
        # Test different compression types
        results = {}
        
        for comp_type in [CompressionType.GZIP, CompressionType.LZMA, CompressionType.ZLIB]:
            try:
                compressed = self.compress_data(data, comp_type)
                ratio = self.get_compression_ratio(len(data), len(compressed))
                results[comp_type] = ratio
            except Exception as e:
                logger.warning(f"Failed to test compression {comp_type}: {e}")
                results[comp_type] = 1.0
        
        # Choose the best ratio (lowest)
        best_type = min(results.keys(), key=lambda k: results[k])
        
        # If compression doesn't help much, don't compress
        if results[best_type] > 0.9:
            return CompressionType.NONE
        
        return best_type


# Global compression instance
_file_compression: Optional[FileCompression] = None


def get_file_compression() -> FileCompression:
    """Get global file compression instance."""
    global _file_compression
    
    if _file_compression is None:
        _file_compression = FileCompression()
    
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    return _file_compression