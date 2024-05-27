"""
Pipeline functions
"""

import os
from XML.createXML import XMLData
from data.dataset import OCRDatasetInstanceSeg
from data.augmentation import ImageRotateXML, ImageCropXML
from model.inference import load_region_model, load_line_model, load_text_recognize_model
from utils.cv_utils import convert_from_cv2_to_image, convert_from_image_to_cv2

import importlib.util
package_name = 'detectron2'
if importlib.util.find_spec(package_name) != None:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.utils.visualizer import ColorMode

def initFolder(opts, infolder, page="page"):
    
    xmldir = os.path.join(infolder, page)
    if not os.path.exists(xmldir):
        os.makedirs(xmldir)
    for fname in os.listdir(infolder):
        infile = os.path.join(infolder, fname)
        if infile.endswith(".jpg") or infile.endswith(".png") or infile.endswith(".jpeg") or infile.endswith(".TIF"):
            basename, _ = os.path.splitext(os.path.basename(infile))
            xmlfile = os.path.join(xmldir, basename + ".xml")
            x = XMLData(opts, infile, xmlfile)
            x.new_xml()
            x.save_xml()

def extractText(opts, infolder, inpage="pageText", outpage="text", outputtype="text"):
    
    dataset_files = [[infolder, os.path.join(infolder, inpage)]]
    dataset = OCRDatasetInstanceSeg(dataset_files, opts.region_classes)
    textdir = os.path.join(infolder, outpage)
    if not os.path.exists(textdir):
        os.makedirs(textdir)
    if outputtype == "text":
        file_path = os.path.join(textdir, "extracted.txt")
    else:    
        file_path = os.path.join(textdir, "extracted.json")
    if outputtype == "text":
        with open(file_path, "w", encoding="utf-8") as f:
            for i in range(dataset.__len__()):
                page = dataset.__getXMLitem__(i)
                f.write("Page: " + str(i+1) + "\n")
                for reg in page:
                    f.write("Region: " + reg['type'] + "\n")
                    for line in reg['lines']:           
                        f.write("Line:" + line['Text'].decode('utf-8') + "\n")
        f.close()
    else:  # outputtype == "json":
        import json
        pages_json = []
        for i in range(dataset.__len__()):
            page = dataset.__getXMLitem__(i)
            reg_json = []
            for reg in page:
                lines_json = []
                for line in reg['lines']:
                    lines_json.append(line['Text'].decode('utf-8'))
                reg_json.append({"region": reg['type'], "lines": lines_json})
            pages_json.append({"PageNumber": i, "PageData": reg_json})
        doc_json = {"pages": pages_json}
        with open(file_path, 'w') as outfile:
            json.dump(doc_json, outfile)

def pipelineTask(opts, task, infolder, inpage="page", outpage=None, tryMerge=False, combine=False, reading_order=True, line_model="mask r-cnn"):
#    print("pipleineTask:", tryMerge)
    if task.lower() in ["text", "json"]:
        if task.lower() == "text":
            extractText(opts, infolder, inpage, outpage, outputtype="text")
        else:
            extractText(opts, infolder, inpage, outpage, outputtype="json")
    else:
        dataset_files = [[infolder, os.path.join(infolder, inpage)]]
        if task.lower() in ["regiondetection", "rd", "region"]:
            dataset = OCRDatasetInstanceSeg(dataset_files, opts.region_classes)
            device = opts.device
            if opts.region_ensemble:
                region_predictor, region_predictor2 = load_region_model(opts, device)
            else:
                region_predictor = load_region_model(opts, device)
        
        elif task in ["linedetection","ld", "line"]:
            dataset = OCRDatasetInstanceSeg(dataset_files, opts.line_classes)
            device = opts.device
            if not opts.package_detectron2:
                raise ModuleNotFoundError()
            line_predictor = load_line_model(opts, device)

        elif task in ["columndetection","cd", "column","rowdetection", "row"]:
            dataset = OCRDatasetInstanceSeg(dataset_files, opts.region_classes)

            if not opts.package_detectron2:
                raise ModuleNotFoundError()

            line_cfg = get_inference_cfg(opts.line_config_file_path, opts.line_checkpoint_url,  len(opts.regions_classes), opts.device)
            line_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            line_predictor = DefaultPredictor(line_cfg)
        elif task in ["update","u", "lineupdate"]:
            dataset = OCRDatasetInstanceSeg(dataset_files, opts.line_classes)
        elif task in ["rt", "recognizetext"]:
            dataset = OCRDatasetInstanceSeg(dataset_files, opts.region_classes)
            if opts.text_recognize.lower() == "htr":
                load_text_recognize_model(opts, opts.device)
        
        for idx in range(dataset.__len__()):
            infile = dataset.__getfullname__(idx)
            fname = os.path.basename(infile)
            print("Starting: {}".format(fname))
            xmldir = os.path.join(os.path.dirname(infile), outpage)
            if not os.path.exists(xmldir):
                os.makedirs(xmldir)
            basename, _ = os.path.splitext(fname)
            xmlfile = os.path.join(xmldir, basename + ".xml")
            if task.lower() in ["regiondetection", "rd", "region"]:
                print("RD called", tryMerge)
                x = XMLData(opts, infile, xmlfile, region_classes=opts.region_classes, tryMerge=tryMerge, reading_order=True)
                x.new_xml()
                if opts.region_ensemble:
                    x._processRegions(opts, dataset, idx, region_predictor,model2=region_predictor2)
                else:
                    x._processRegions(opts, dataset, idx, region_predictor)
            elif task in ["linedetection","ld", "line"]:
                if line_model == "mask r-cnn":  
                    x = XMLData(opts, infile, xmlfile, 
                                line_classes=opts.line_classes, line_model="mask r-cnn", tryMerge=tryMerge, reading_order=True)
                elif line_model == "kraken":
                    x = XMLData(opts, infile, xmlfile, 
                                line_classes=opts.line_classes, line_model="kraken", tryMerge=False, reading_order=True)
                x.new_xml()
                x._processLines(opts, dataset, idx, line_predictor)
            elif task in ["columndetection","cd", "column"]:
                x = XMLData(opts, infile, xmlfile, 
                           line_classes=opts.line_classes, line_model="mask r-cnn", tryMerge=tryMerge, reading_order=True)
                x.new_xml()
                x._processColumns(opts, dataset, idx, line_predictor)
            elif task in ["rowdetection", "row"]:
                x = XMLData(opts, infile, xmlfile, 
                            line_classes=opts.line_classes, line_model="mask r-cnn", tryMerge=tryMerge, reading_order=True)
                x.new_xml()
                x._processRows(opts, dataset, idx, line_predictor)
            elif task in ["lineupdate", "lu", "update"]:
                x = XMLData(opts, infile, xmlfile, 
                            line_classes=opts.line_classes, tryMerge=False, reading_order=True, combine=combine)
                x.new_xml()
                x._updateLines(opts, dataset, idx)
            elif task in ["textrecognition", "tr", "recognize", "rt"]:
                x = XMLData(opts, infile, xmlfile, text_recognition=True, 
                            line_classes=opts.line_classes, reading_order=True)
                x.new_xml()
                x._processTextLines(opts, dataset, idx)
            
            x.save_xml()

def transformXML(opts, task, infolder, outfolder, inpage="page", outpage="page", angle=0.0, crop=(0,0)):
    dataset_files = [[infolder, os.path.join(infolder, inpage)]]
    dataset = OCRDatasetInstanceSeg(dataset_files, opts.region_classes)

    for i in range(dataset.__len__()):
        basename,_ = os.path.splitext(os.path.basename(dataset.__getfullname__(i)))
        if task.lower() in ["c", "crop"]:
            t_img, page = ImageCropXML(dataset,i, crop)
        elif task.lower() in ["r", "rotate"]:
            t_img, page = ImageRotateXML(dataset, i, angle)

        outfile = os.path.join(outfolder, basename + ".jpg")
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        t_img = convert_from_cv2_to_image(t_img)
        t_img = t_img.save(outfile)  
        xmlfile = os.path.join(os.path.join(outfolder, outpage), basename + ".xml")
        if not os.path.exists(os.path.dirname(xmlfile)):
            os.makedirs(os.path.dirname(xmlfile))
    
        x = XMLData(opts, outfile, xmlfile)
        x.new_xml()
        x._modifyCoords(opts, page)
        x.save_xml()
