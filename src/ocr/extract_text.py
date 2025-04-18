from paddleocr import PaddleOCR
import os

ocr_en = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_image(image_path):
    result = ocr_en.ocr(image_path, cls=True)
    return " ".join([line[1][0] for block in result for line in block]).strip()

def extract_text_from_folder(folder):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd
    from tqdm import tqdm

    png_files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    
    def process_image(filename):
        try:
            path = os.path.join(folder, filename)
            text = extract_text_from_image(path)
            return {"filename": filename, "comment": text or None}
        except:
            return {"filename": filename, "comment": None}

    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_image, f) for f in png_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv("data/processed/extracted_memes_english_only.csv", index=False)
