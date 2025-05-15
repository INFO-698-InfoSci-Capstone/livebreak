# Text-to-Speech with AI Anchor using SadTalker

This project demonstrates a pipeline that converts news articles or any web content into speech using Python libraries and generates a talking-head AI anchor video using [SadTalker](https://github.com/Winfredy/SadTalker). The output is a lifelike avatar reading the extracted text aloud.

## Features

- Extracts text from a given URL using `BeautifulSoup`
- Converts text to speech using `pyttsx3`
- Uses `SadTalker` to animate a static face image with the generated voice
- Outputs a complete AI-generated news anchor video

## Technologies Used

- Python 3.8+
- [pyttsx3](https://pypi.org/project/pyttsx3/) for offline Text-to-Speech
- BeautifulSoup for web scraping
- FFmpeg for audio/video processing
- SadTalker for AI-driven talking-head video generation
- OpenCV, NumPy, Matplotlib

## Setup Instructions

1. **Clone SadTalker**

   ```bash
   git clone https://github.com/Winfredy/SadTalker
   cd SadTalker
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   pip install pyttsx3 beautifulsoup4 requests
   ```

3. **Download Pretrained Models**

   ```bash
   bash scripts/download_models.sh
   ```

4. **Prepare Input Files**
   - Upload your **face image** to `examples/source_image/`
   - The script will save the generated **audio** as `news_audio.wav` and place it into `examples/driven_audio/`

## How to Run

```python
# Step 1: Extract text from a news URL
# Step 2: Convert text to speech using pyttsx3
# Step 3: Save audio file as news_audio.wav
# Step 4: Copy image and audio into SadTalker inputs
# Step 5: Run SadTalker inference
```

## Output

- The final video is saved in the `./results/` folder.
- It features your input face image animated with lip-sync and head motion driven by the input speech.

## Credits

- [SadTalker](https://github.com/Winfredy/SadTalker) – for real-time talking-head generation
- [pyttsx3](https://pypi.org/project/pyttsx3/) – for offline TTS
- Livebreak News (for content extraction)
