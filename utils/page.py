"""
Page utilities
"""
import numpy as np
import shapely
from shapely.ops import linemerge, unary_union, polygonize

from shapely.geometry import LineString, Polygon
from shapely.errors import TopologicalError
import math
from post_processing.polygon import make_valid

def createBaseline(opts, textline):
    l_xmax, l_ymax = textline.max(axis=0)
    l_xmin, l_ymin = textline.min(axis=0)

    line = LineString([(int((l_xmax + l_xmin) / 2), 0), (int((l_xmax + l_xmin) / 2), l_ymax + 10)])
    p = Polygon(textline)
    merged = linemerge([p.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    poly_list = list(polygons)
    x0_min_1, y0_min_1, x0_max_2, y0_max_2 = poly_list[0].bounds
    x1_min_1, y1_min_1, x1_max_2, y1_max_2 = poly_list[1].bounds

    if x0_min_1 < x0_min_1:
        baseline = np.array([[x0_min_1, y0_max_2], [x0_max_2, y0_max_2], [x1_max_2, y1_max_2]]).astype(int)
    else:
        baseline = np.array([[x1_min_1, y1_max_2], [x1_max_2, y1_max_2], [x0_max_2, y0_max_2]]).astype(int)
    return baseline


def createBaseline_new(opts, textline):
    l_xmax, l_ymax = textline.max(axis=0)
    l_xmin, l_ymin = textline.min(axis=0)
    base_offset = int((l_ymax - l_ymin + 1) * opts.baseline_offset_multiplier)
    baseline = []
    for i in range(math.ceil((l_xmax - l_xmin + 1) / opts.baseline_sample_size)):
        x_lower, x_upper = int(l_xmin + i * opts.baseline_sample_size), int(
            l_xmin + (i + 1) * opts.baseline_sample_size) - 1
        line1 = LineString([(x_lower, 0), (x_lower, l_ymax + opts.default_y_offset)])
        line2 = LineString([(x_upper - 1, 0), (x_upper - 1, l_ymax + opts.default_y_offset)])
        p = Polygon(textline)
        merged = linemerge([p.boundary, line1, line2])
        borders = unary_union(merged)
        polygons = polygonize(borders)
        poly_list = list(polygons)
        for j in range(len(poly_list)):
            x1, y1, x2, y2 = [int(element) for element in poly_list[j].bounds]
            if (x1 >= x_lower) and (x2 <= x_upper):
                baseline.append((x1, y2))
                baseline.append((x2, y2))
    baseline = [(element[0], element[1] - base_offset) for element in baseline]
    return np.array(baseline)


def createBaseline_old2(textline):
    l_xmax, l_ymax = textline.max(axis=0)
    l_xmin, l_ymin = textline.min(axis=0)
    return np.array([[l_xmin, l_ymax], [l_xmax, l_ymax]])


def createBbox(textline):
    l_xmax, l_ymax = textline.max(axis=0)
    l_xmin, l_ymin = textline.min(axis=0)
    return [(l_xmin, l_ymin), (l_xmax, l_ymax)]


def get_lines_area(lines, width, height):
    if len(lines) > 0:
        a_xmax, a_ymax = -np.inf, -np.inf
        a_xmin, a_ymin = np.inf, np.inf
    else:
        a_xmax, a_ymax = width, height
        a_xmin, a_ymin = 0, 0

    for line in lines:
        l_xmax, l_ymax = line['Textline'].max(axis=0)
        l_xmin, l_ymin = line['Textline'].min(axis=0)
        if (l_xmax > a_xmax): a_xmax = l_xmax
        if (l_ymax > a_ymax): a_ymax = l_ymax
        if (l_xmin < a_xmin): a_xmin = l_xmin
        if (l_ymin < a_ymin): a_ymin = l_ymin
    return [(a_xmin, a_ymin), (a_xmax, a_ymax)]


def map_lines_to_region(opts, region, lines):
    r_xmax, r_ymax = region['polygon'].max(axis=0)
    r_xmin, r_ymin = region['polygon'].min(axis=0)
    newreg = {}
    newreg['id'] = region['id']
    newreg['type'] = region['type']
    newreg['polygon'] = region['polygon']
    newreg['bbox'] = region['bbox']
    reg_poly = make_valid(Polygon(region['polygon']))

    reglines = []
    for line in lines:
        regline = {}
        l_xmax, l_ymax = line['Textline'].max(axis=0)
        l_xmin, l_ymin = line['Textline'].min(axis=0)

        if ((r_xmin - opts.line_boundary_margin) < l_xmin) and (l_xmin < (r_xmax + opts.line_boundary_margin)) and \
                ((r_ymin - opts.line_boundary_margin) < l_ymin) and (l_ymin < (r_ymax + opts.line_boundary_margin)) and \
                ((r_xmin - opts.line_boundary_margin) < l_xmax) and (l_xmax < (r_xmax + opts.line_boundary_margin)) and \
                ((r_ymin - opts.line_boundary_margin) < l_ymax) and (l_ymax < (r_ymax + opts.line_boundary_margin)):
            baseline = createBaseline_new(opts, line['Textline'])
            bbox = createBbox(line['Textline'])
            regline = {'id': line['id'],
                       'bbox': bbox,
                       'Textline': line['Textline'],
                       'Baseline': baseline}
        else:
            line_poly = Polygon(line['Textline'])
            try:
                line_intersect = reg_poly.intersection(line_poly)
                if reg_poly.intersects(line_poly) and type(line_intersect) == shapely.geometry.polygon.Polygon:
                    detected_line = np.array(list(line_intersect.exterior.coords)).astype(int)
                    d_xmax, d_ymax = detected_line.max(axis=0)
                    d_xmin, d_ymin = detected_line.min(axis=0)
                    if (d_xmax - d_xmin) > opts.min_line_width and (d_ymax - d_ymin) > opts.min_line_heigth:
                        baseline = createBaseline_new(opts, detected_line)
                        bbox = createBbox(line['Textline'])
                        regline = {'id': line['id'] + "_1",
                                   'bbox': bbox,
                                   'Textline': detected_line,
                                   'Baseline': baseline}
                    else:
                        print("Small line rejected: {} {} {} {}".format(d_xmin, d_ymin, d_xmax, d_ymax))
                        regline = {}
                else:
                    regline = {}
            except:
                print("Invalid intersection++++++")
                print("Region:", [(r_xmin, r_ymin), (r_xmax, r_ymax)])
                print("Line:", [(l_xmin, l_ymin), (l_xmax, l_ymax)])
        #                drawLinePolygon(image_path,region['polygon'])
        #                drawLinePolygon(image_path, line['Textline'])

        if len(regline) > 0:
            reglines.append(regline)
    newreg['lines'] = reglines
#    print("Regions and lines:", len(reglines))
#    for lb in reglines: print(lb)
    return newreg


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return int(float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]))

    div = det(xdiff, ydiff)
    if div == 0:
        print('lines do not intersect')
        return
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def splitRegion(region, lineA, lineB):
    l_xmax, l_ymax = region.max(axis=0)
    l_xmin, l_ymin = region.min(axis=0)
    line1 = LineString(lineA)
    line2 = LineString(lineB)
    p = Polygon(region)
    merged = linemerge([p.boundary, line1, line2])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    poly_list = list(polygons)
    poly = []
    for j in range(len(poly_list)):
        area = []
        x1, y1, x2, y2 = [int(element) for element in poly_list[j].bounds]
        #        print(x1, y1, x2, y2)
        area.append((x1, y1))
        area.append((x2, y1))
        area.append((x2, y2))
        area.append((x1, y2))
        print(area)
        poly.append(area)
    return np.array(poly)
