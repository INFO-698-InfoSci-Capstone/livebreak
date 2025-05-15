# Integrated OCR System

This repository contains an integrated OCR (Optical Character Recognition) system that combines multiple OCR engines to handle both handwritten and printed text recognition.

## Features

- **Dual OCR Engine System**:
  - **TrOCR** for handwritten text recognition
  - **Tesseract OCR** for printed text recognition
- **Auto-detection** of text type (handwritten vs. printed)
- **Visualization** of OCR results
- **Batch processing** of multiple images

## Requirements

### Python Dependencies

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.11+
- OpenCV (cv2)
- Pillow (PIL)
- NumPy
- Matplotlib
- pytesseract

### External Dependencies

- **Tesseract OCR** - For printed text recognition

## Installation

### 1. Install Python Dependencies

Install the required Python packages:
- torch and torchvision
- transformers
- opencv-python
- pillow
- numpy
- matplotlib
- pytesseract

We recommend using a virtual environment for this installation.

### 2. Install Tesseract OCR

#### On Windows:
1. Download the installer from UB Mannheim's GitHub repository
2. Run the installer and follow the installation instructions
3. Add Tesseract to your PATH environment variable

#### On macOS:
Use Homebrew to install Tesseract

#### On Ubuntu/Debian:
Install using apt package manager

### 3. Verify Installation

Verify that Tesseract, PyTorch, and the Transformers library are properly installed and working.

## Running the System

### Setting Up the Project

1. Clone or download this repository to your local machine
2. Navigate to the project directory

### Running with Default Settings

To process images with the default settings:

1. Place your images in the "Images" directory
2. Run the script:
   ```
   python Integrated_ocr.py
   ```

### Customizing Image Paths

If your images are in a different location, modify the `image_paths` list in the `Integrated_ocr.py` file:

```python
image_paths = [
    "path/to/your/image1.jpg",
    "path/to/your/image2.jpg",
    "path/to/your/image3.jpg"
]
```

### Changing OCR Model

To use a different TrOCR model for handwritten text recognition, modify the model name in `integrated_ocr.py`:

```python
trocr_model_name = "microsoft/trocr-base-handwritten"  # Using base model instead of large
```

### Output Configuration

By default, the system will:
- Save text extraction results in the "output" directory
- Generate visualizations if the `visualize` variable is set to `True`

You can modify these settings in the `Integrated_ocr.py` file.

## Usage

### Basic Usage

Initialize the OCR system and process images to extract text.

### Processing with Visualization

Process images and generate visualizations of the OCR results.

### Force Specific OCR Method

Force the system to use a specific OCR method (handwritten or printed) when needed.

### Batch Processing

Process multiple images in batch mode.

## Output

The system produces several types of output:

### Text Extraction Results

When processing an image, the system returns a dictionary containing:
- `image_path`: Path to the processed image
- `text_type`: Detected text type ("handwritten" or "printed")
- `method_used`: OCR method that was applied ("TrOCR" or "Tesseract")
- `full_text`: The extracted text content

### Text Files

For each processed image, a text file is generated containing:
- Header information (text type and method used)
- The full extracted text

### Visualizations

The system can generate visualization images showing:
- The original image with a colored border indicating text type
- The extracted text displayed below the image
- Method information

## Directory Structure

- Integrated_ocr.py: Main OCR system class
- output/: Output directory for results
- Images/: Directory containing input images

## Model Options

The system supports different TrOCR models for handwritten text recognition:

- microsoft/trocr-large-handwritten (default, best quality)
- microsoft/trocr-base-handwritten (faster, less memory usage)

## Notes

- The first time you use a TrOCR model, it will be downloaded automatically.
- GPU acceleration is used when available, significantly improving TrOCR performance.
- The text type detection is heuristic-based and may not always be accurate.
- Tesseract works best with clearly printed text; document preprocessing may improve results.

## Troubleshooting

- If you receive an error about Tesseract not being found, ensure it's properly installed and added to your PATH.
- For CUDA out-of-memory errors with TrOCR, try using the base model instead of large.
- If OCR results are poor, try preprocessing your images (increasing contrast, removing noise) before processing.


