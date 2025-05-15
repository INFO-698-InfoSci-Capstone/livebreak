import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

class IntegratedOCRSystem:
    """Combined OCR system that handles both handwritten text (TrOCR) and printed text (Tesseract)."""
    
    def __init__(self, trocr_model_name: str = "microsoft/trocr-large-handwritten"):
        """
        Initialize the integrated OCR system.
        
        Args:
            trocr_model_name: The TrOCR model to use for handwritten text OCR. Options include:
                - "microsoft/trocr-large-handwritten" (default, best for handwritten text)
                - "microsoft/trocr-base-handwritten" (lighter model for handwritten text)
        """
        # Initialize TrOCR for handwritten text
        print(f"Loading TrOCR model for handwritten text: {trocr_model_name}")
        self.processor = TrOCRProcessor.from_pretrained(trocr_model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Check if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR is available for printed text")
        except Exception as e:
            print(f"Warning: Tesseract OCR not properly installed: {e}")
            print("Printed text recognition may not work correctly")
    
    def handwritten_text_recognition(self, image_path: str) -> str:
        """
        Process handwritten text image with TrOCR.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Recognized text
        """
        try:
            image = Image.open(image_path)
            
            # Process image with TrOCR processor
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text with more lenient parameters
            generated_ids = self.model.generate(
                pixel_values,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode the generated ids to text
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return predicted_text
            
        except Exception as e:
            raise ValueError(f"Error processing handwritten text in image {image_path}: {str(e)}")
    
    def printed_text_recognition(self, image_path: str) -> str:
        """
        Process printed text image with Tesseract OCR.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Recognized text
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding (make text stand out)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Create a temporary file for processed image
            temp_file = "temp_processed.jpg"
            cv2.imwrite(temp_file, thresh)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(Image.open(temp_file))
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return text
            
        except Exception as e:
            raise ValueError(f"Error processing printed text in image {image_path}: {str(e)}")
    
    def detect_text_type(self, image_path: str) -> str:
        """
        Detect whether the image contains printed text or handwritten text.
        This is a simple heuristic and might not be perfect.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Text type: "printed" or "handwritten"
        """
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate the number of black pixels (text)
        black_pixels = np.sum(binary == 0)
        
        # Calculate the variance in stroke width (a simple method)
        # Use edge detection to find text contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate width variation of contours
        widths = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Filter out noise
                widths.append(w)
        
        # High variance often indicates handwritten text
        width_variance = np.var(widths) if widths else 0
        
        # Use filename hint if available
        filename = os.path.basename(image_path).lower()
        if "handwritten" in filename or "hand" in filename:
            return "handwritten"
        elif "printed" in filename or "print" in filename:
            return "printed"
        
        # Use heuristic based on variance and other factors
        if width_variance > 100:
            return "handwritten"
        else:
            return "printed"
    
    def process_image(self, image_path: str, force_method: str = None) -> dict:
        """
        Process an image and extract text using the appropriate method.
        
        Args:
            image_path: Path to the image file
            force_method: Force a specific method ("handwritten" or "printed")
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not os.path.isfile(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # Determine text type if not forced
        if force_method:
            text_type = force_method
        else:
            text_type = self.detect_text_type(image_path)
        
        # Extract text using appropriate method
        if text_type == "handwritten":
            text = self.handwritten_text_recognition(image_path)
            method = "TrOCR"
        else:
            text = self.printed_text_recognition(image_path)
            method = "Tesseract"
        
        result = {
            "image_path": image_path,
            "text_type": text_type,
            "method_used": method,
            "full_text": text
        }
        
        return result
    
    def visualize_results(self, image_path: str, result: dict) -> np.ndarray:
        """
        Visualize the OCR results on the image.
        
        Args:
            image_path: Path to the original image
            result: OCR result dictionary from process_image
            
        Returns:
            Image with visualized results
        """
        # Read original image
        image = cv2.imread(image_path)
        if image is None:
            # Try with PIL and convert
            try:
                pil_img = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                raise ValueError(f"Could not read image at {image_path}")
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Get image dimensions
        h, w = vis_image.shape[:2]
        
        # Add a colored border based on text type
        border_color = (0, 150, 255) if result["text_type"] == "handwritten" else (0, 255, 0)
        border_width = 10
        vis_image = cv2.copyMakeBorder(vis_image, border_width, border_width, border_width, border_width, 
                                       cv2.BORDER_CONSTANT, value=border_color)
        
        # Create a white background for the text
        text_height = min(150, h // 3)  # Text area height
        text_area = np.ones((text_height, w + 2*border_width, 3), dtype=np.uint8) * 255
        
        # Add text type and method info
        method_text = f"Text Type: {result['text_type'].upper()} | Method: {result['method_used']}"
        cv2.putText(text_area, method_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
        
        # Add header for extracted text
        cv2.putText(text_area, "Extracted Text:", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Split long text into multiple lines
        full_text = result["full_text"]
        max_chars_per_line = (w + 2*border_width) // 10  # Estimate chars that fit on one line
        text_lines = []
        
        for i in range(0, len(full_text), max_chars_per_line):
            text_lines.append(full_text[i:i + max_chars_per_line])
        
        # Display up to 3 lines of text
        font_scale = 1.5
        for i, line in enumerate(text_lines[:3]):
            y_pos = 85 + (i * 25)  # Position text with spacing between lines
            cv2.putText(text_area, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 200), 2)
        
        # Combine the visualization with the text display
        combined = np.vstack([vis_image, text_area])
        
        return combined


def main():
    # Direct specification of image paths
    image_paths = [
        "OCR based text extraction/Images/PXL_20250421_184306746~3.jpg",  
        "OCR based text extraction/Images/PXL_20250421_184207617~2.jpg",       
        "OCR based text extraction/Images/PXL_20250421_184207617~5.jpg",
        "OCR based text extraction/Images/printed_text_1.jpg"  
    ]
    
    # Initialize the integrated OCR system
    trocr_model_name = "microsoft/trocr-large-handwritten"  # For handwritten text
    ocr_system = IntegratedOCRSystem(trocr_model_name=trocr_model_name)
    
    # Other parameters
    output_dir = "output"
    visualize = True
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image
    for img_path in image_paths:
        print(f"\nProcessing {img_path}...")
        
        try:
            # Verify file exists
            if not os.path.isfile(img_path):
                print(f"Error: File {img_path} does not exist")
                continue
                
            # Process the image - auto-detect text type
            result = ocr_system.process_image(img_path)
            
            # Print the extracted text and method used
            print("\n" + "="*50)
            print(f"TEXT TYPE: {result['text_type'].upper()} | METHOD: {result['method_used']}")
            print("-"*50)
            print(result["full_text"])
            print("="*50 + "\n")
            
            # Save text output
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            text_output_path = os.path.join(output_dir, f"{base_name}_text.txt")
            
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(f"Text Type: {result['text_type']}\n")
                f.write(f"Method Used: {result['method_used']}\n\n")
                f.write(result["full_text"])
            
            print(f"Text saved to {text_output_path}")
            
            # Visualize
            if visualize:
                vis_image = ocr_system.visualize_results(img_path, result)
                vis_output_path = os.path.join(output_dir, f"{base_name}_visualized.jpg")
                cv2.imwrite(vis_output_path, vis_image)
                print(f"Visualization saved to {vis_output_path}")
                
                # Optional: Display the image if in interactive environment
                try:
                    plt.figure(figsize=(12, 10))
                    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.title(f"OCR Results: {base_name}")
                    plt.show()
                except Exception as e:
                    print(f"Could not display image: {e}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()