#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
sys.path.append('/home/arive/HumePipeline')

from config.options import Options

from data.dataset import OCRDatasetInstanceSeg
from pipeline.engine import initFolder, pipelineTask, transformXML

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

    parser = argparse.ArgumentParser(description='Data augmentation for XML files')

    parser.add_argument('--task', dest='executeTask', type=str, 
                        help='Describes the task to be performed')
    parser.add_argument('--indir', dest='infolder', type=str, 
                        help='Directory of images')
    parser.add_argument('--inpage', dest='inpage', type=str, 
                        help='Input directory for PAGE XML files')
    parser.add_argument('--outdir', dest='outfolder', type=str, 
                        help='Output directory for images')
    parser.add_argument('--outpage', dest='outpage', type=str, 
                        help='Output directory for PAGE XML files')
    parser.add_argument('--config', dest='cfgfile', type=str, 
                        help='Location of configuration file')
    parser.add_argument('--angle', dest='angle', type=str, 
                        help='Rotation angled')
    parser.add_argument('--cropx', dest='cropx', type=str, 
                        help='How many pixels cropped from x-axis')
    parser.add_argument('--cropy', dest='cropy', type=str, 
                        help='How many pixels cropped for y-axis')

    args = parser.parse_args()

#    if args.executeTask != None:
#        print(args.executeTask)
#    if args.infolder != None:
#        print(args.infolder)
#    if args.inpage != None:
#        print(args.inpage)
#    if args.outfolder != None:
#        print(args.outfolder)
#    if args.outpage != None:
#        print(args.outpage)
#    if args.cfgfile != None:
#        print(args.cfgfile)
#    if args.cropx != None:
#        print(args.cropx)
#    if args.cropy != None:
#        print(args.cropy)
#    if args.angle != None:
#        print(args.angle)

    if args.cfgfile != None:
         opts =  Options(args.cfgfile)

    if args.executeTask != None:    
        if args.inpage == None:
            inpage = "page"
        else:
            inpage = args.inpage
        if args.outpage == None:
            outpage = "page"
        else:
            outpage = args.outpage
        if args.angle != None:
            angle = float(args.angle)
        if args.cropx != None:
            cropx = int(args.cropx)
        if args.cropy != None:
            cropy = int(args.cropy)

        if args.infolder != None and args.outfolder != None:
            RO_groups = opts.RO_region_groups
            if args.executeTask.lower() == "rotate" or args.executeTask.lower() == "r":
                transformXML(opts, "rotate", args.infolder, args.outfolder, inpage=inpage, outpage=outpage, angle=angle)
            elif args.executeTask.lower() == "crop" or args.executeTask.lower() == "c":
                transformXML(opts, "crop", args.infolder, args.outfolder, inpage=inpage, outpage=outpage, crop=(cropx,cropy))
   
            else:
                print("Task not recognized")
