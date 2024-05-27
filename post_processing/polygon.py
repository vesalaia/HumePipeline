"""
Postprocessing
"""
import shapely
from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon
from shapely.errors import TopologicalError
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos

def make_valid(ob):

    if ob.is_valid:
        return ob
    try:
        ret = geom_factory(lgeos.GEOSMakeValid(ob._geom))
    except:
        ret = ob
    return ret

