#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
sys.path.append('/home/arive/HumePipeline')

from config.options import Options

from data.dataset import OCRDatasetInstanceSeg
from pipeline.engine import initFolder, extractText, pipelineTask

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#
# Main progranm
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Utilities for OCR end-2-end pipeline')

    parser.add_argument('--task', dest='executeTask', type=str, 
                        help='Describes the task to be performed')
    parser.add_argument('--dir', dest='infolder', type=str, 
                        help='Directory of images')
    parser.add_argument('--inpage', dest='inpage', type=str, 
                        help='Input directory for PAGE XML files')
    parser.add_argument('--outpage', dest='outpage', type=str, 
                        help='Output directory for PAGE XML files')
    parser.add_argument('--config', dest='cfgfile', type=str, 
                        help='Location of configuration file')
    parser.add_argument('--linedir', dest='linedir', type=str, 
                        help='Input directory of Line images')
    parser.add_argument('--linpage', dest='linpage', type=str, 
                        help='Input directory of Line image PAGE XML files')
    parser.add_argument('--regiondir', dest='regiondir', type=str, 
                        help='Input directory of region images')
    parser.add_argument('--rinpage', dest='rinpage', type=str, 
                        help='Input directory of Region image PAGE XML files')
    parser.add_argument('--merge',  dest='merge', type=str, help="Try overlapping region merge")
    parser.add_argument('--ro',  dest='ro', type=str, help="Try overlapping region mergeCheck the reading order of regions")
    parser.add_argument('--combine', dest='combine', type=str, help="Try combining lines")

  #  parser.add_argument('--merge', dest='tryMerge',type=str2bool, nargs='?', default=False, 
  #                      help='Try overlapping region merge')
  #  parser.add_argument('--combine', dest='combine', type=str2bool, nargs='?', default=False, 
  #                      help='Combine small lines if possible')
  #  parser.add_argument('--ro', dest='reading_order', type=str2bool, nargs='?', default=False, 
  #                      help='Check the reading order of regions')
    args = parser.parse_args()

    if args.executeTask != None:
        print(args.executeTask)
    if args.infolder != None:
        print(args.infolder)
    if args.inpage != None:
        print(args.inpage)
    if args.outpage != None:
        print(args.outpage)
    if args.cfgfile != None:
        print(args.cfgfile)
    if args.linedir != None:
        print(args.linedir)
    if args.linpage != None:
        print(args.linpage)
    if args.regiondir != None:
        print(args.regiondir)
    if args.rinpage != None:
        print(args.rinpage)
    if args.merge:
        tryMerge = str2bool(args.merge)
        print("Merge:", args.merge, str2bool(args.merge))
    else:
        tryMerge = False
    if args.combine:
        combine = str2bool(args.combine)
        print("Combine:", args.combine, str2bool(args.combine))
    else:
         combine = False
    if args.ro:
        reading_order = str2bool(args.ro)
        print("RO:", args.ro, str2bool(args.ro))
    else:
        reading_order = False
    if args.cfgfile != None:
         opts =  Options(args.cfgfile)

    
    if args.executeTask != None:    
        if args.executeTask.lower() == "init" or args.executeTask.lower() == "i":
            if args.inpage == None:
                inpage = "page"
            else:
                inpage = args.inpage
            if args.infolder != None:
                initFolder(opts, args.infolder, inpage) 
        elif args.executeTask.lower() in ["text", "json"]:
            if args.inpage == None:
                inpage = "pageText"
            else:
                inpage = args.inpage
            if args.outpage == None:
                outpage = "text"
            else:
                outpage = args.outpage
            if args.infolder != None:
                pipelineTask(opts, args.executeTask,args.infolder, inpage, outpage) 
        elif args.executeTask.lower() == "detectregions" or args.executeTask.lower() == "dr":
            if args.inpage == None:
                inpage = "page"
            else:
                inpage = args.inpage
            if args.outpage == None:
                outpage = "pageRD"
            else:
                outpage = args.outpage
  #          if args.merge == None:
  #              tryMerge = False
  #          else:
  #              tryMerge = True
            if args.infolder != None:
                RO_groups = opts.RO_region_groups
                print("Merge:", tryMerge)
                pipelineTask(opts, "rd", args.infolder, inpage=inpage, outpage=outpage, tryMerge=tryMerge, 
                             reading_order=reading_order)
        elif args.executeTask.lower() == "detectlines" or args.executeTask.lower() == "dl":
            if args.inpage == None:
                inpage = "pageRD"
            else:
                inpage = args.inpage
            if args.outpage == None:
                outpage = "pageLD"
            else:
                outpage = args.outpage
   #         if args.merge == None:
   #             tryMerge = False
   #         else:
   #             tryMerge = True
            if args.infolder != None:
                RO_groups = opts.RO_region_groups
                one_page_per_image = True
                pipelineTask(opts, "ld", args.infolder, inpage=inpage, outpage=outpage, tryMerge=tryMerge,
                            reading_order=reading_order, line_model="mask r-cnn")
        elif args.executeTask.lower() == "recognizetext" or args.executeTask.lower() == "rt":
            if args.inpage == None:
                inpage = "pageLD"
            else:
                inpage = args.inpage
            if args.outpage == None:
                outpage = "pageText"
            else:
                outpage = args.outpage
            if args.infolder != None:
                RO_groups = opts.RO_region_groups
                DEBUG=False
                pipelineTask(opts, "rt", args.infolder, inpage=inpage, outpage=outpage, 
                             reading_order=reading_order)
        elif args.executeTask.lower() == "update" or args.executeTask.lower() == "u":
            if args.inpage == None:
                inpage = "pageLD"
            else:
                inpage = args.inpage
            if args.outpage == None:
                outpage = "pageU"
            else:
                outpage = args.outpage
            if args.combine:
                combine = True
            else:
                combine = False
            if args.infolder != None:
                RO_groups = opts.RO_line_groups
                DEBUG=True
                opts.debug = True
                pipelineTask(opts, "update", args.infolder, inpage=inpage, outpage=outpage, 
                             reading_order=reading_order, combine=combine)
        
        else:
             print("Task not recognized")
