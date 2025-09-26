import os
from PIL import Image
import argparse
import glob

def convert_tiff_to_png(input_path, output_path=None, quality=95):
    """
    Convert a single TIFF file to PNG format
    
    Args:
        input_path (str): Path to the input TIFF file
        output_path (str): Path for the output PNG file (optional)
        quality (int): Compression quality (not used for PNG, but kept for consistency)
    """
    try:
        # Open the TIFF image
        with Image.open(input_path) as img:
            # If no output path specified, create one
            if output_path is None:
                base_name = os.path.splitext(input_path)[0]
                output_path = f"{base_name}.png"
            
            # Convert and save as PNG
            # PNG supports transparency, so we preserve it if present
            if img.mode in ('RGBA', 'LA'):
                img.save(output_path, 'PNG', optimize=True)
            else:
                # Convert to RGB if needed for better compatibility
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'PNG', optimize=True)
                
        print(f"Successfully converted: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def batch_convert_tiff_to_png(input_directory, output_directory=None):
    """
    Convert all TIFF files in a directory to PNG format
    
    Args:
        input_directory (str): Directory containing TIFF files
        output_directory (str): Directory for output PNG files (optional)
    """
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Find all TIFF files (case insensitive)
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tiff_files = []
    
    for pattern in tiff_patterns:
        tiff_files.extend(glob.glob(os.path.join(input_directory, pattern)))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_directory}")
        return
    
    successful = 0
    failed = 0
    
    for tiff_file in tiff_files:
        base_name = os.path.splitext(os.path.basename(tiff_file))[0]
        
        if output_directory:
            output_path = os.path.join(output_directory, f"{base_name}.png")
        else:
            output_path = os.path.join(input_directory, f"{base_name}.png")
        
        if convert_tiff_to_png(tiff_file, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion complete! Success: {successful}, Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description='Convert TIFF files to PNG format')
    parser.add_argument('input', help='Input TIFF file or directory')
    parser.add_argument('-o', '--output', help='Output PNG file or directory')
    parser.add_argument('-b', '--batch', action='store_true', 
                       help='Batch convert all TIFF files in input directory')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        batch_convert_tiff_to_png(args.input, args.output)
    else:
        convert_tiff_to_png(args.input, args.output)

if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        print("TIFF to PNG Converter")
        print("\nExample usage:")
        print("python tiff_to_png.py image.tif")
        print("python tiff_to_png.py image.tif -o converted.png")
        print("python tiff_to_png.py /path/to/tiff/folder -b")
        print("python tiff_to_png.py /path/to/tiff/folder -o /path/to/output/folder -b")
        
        # Simple interactive mode
        input_file = input("\nEnter TIFF file path (or press Enter to exit): ").strip()
        if input_file and os.path.exists(input_file):
            convert_tiff_to_png(input_file)
        elif input_file:
            print("File not found!")
    else:
        main()