"""
Text recognition utilities
"""
import pytesseract as pyt

from PIL import Image

def TRImage2Text(opts, img, model=None, processor=None):
    if opts.text_recognize_engine == "tesseract":
        try:
            text = pyt.image_to_string(img, config=opts.text_recognize_custom_config,lang=opts.text_recognize_tesseract_language)
            return text
        except:
            return " "
    elif opts.text_recognize_engine == "trocr":
        pixel_values = processor(img, return_tensors='pt').pixel_values.to(opts.device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
         
    else:
         raise NotImplementedError()

def TRline2Text(opts, imgfile, coords, model=None, processor=None):
    img = Image.open(imgfile).convert("RGB")
    w, h = img.size
    x1, y1 = coords.min(axis=0)
    x2, y2 = coords.max(axis=0)
    if (x2>x1) and (y2>y1):
        if opts.DEBUG: print('##  box=[{},{},{},{}]'.format(x1,y1,x2,y2))
        cropped_image = img.crop((x1,y1, x2,y2))
        if opts.text_recognize_engine.lower() == "tesseract":
            text = TRImage2Text(opts, cropped_image).split("\n")
            ltext = " "
            for l in text:
                if len(l) > len(ltext):
                    ltext = l
            ltext.replace("\n"," ")
            if opts.DEBUG: print("Lines:",len(text), text, len(ltext), ltext)
            return ltext
        elif opts.text_recognize_engine.lower() == "trocr":
            text = TRImage2Text(opts, cropped_image, model, processor)
            if opts.DEBUG: print("Lines:",len(text), text)
            return text
            
