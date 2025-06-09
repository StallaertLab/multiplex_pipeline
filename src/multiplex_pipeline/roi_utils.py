import os
import pickle as pkl
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def pre_select_objects(masks, im, min_area, max_area, min_iou, min_stability, min_int):
    '''
    Function to pre-select objects based on area, iou, stability and intensity.
    
    Args:
        masks (list): The list of masks.
        min_area (int): The minimum area of the object.
        max_area (int): The maximum area of the object.
        min_iou (float): The minimum iou of the object.
        min_stability (float): The minimum stability of the object.
        min_int (float): The minimum intensity of the object.
    Returns:
        list: The filtered list of masks.
    '''

    masks_filtered = []

    for item in masks:
    
        bbox_area =  item['bbox'][2]*item['bbox'][3]

        if ((bbox_area > min_area) and (bbox_area < max_area) and (item['predicted_iou'] > min_iou) and (item['stability_score'] > min_stability)):   
            
            rect = xywh_to_corners(item['bbox']).astype(int)

            mean_int = im[rect[0,0]:rect[2,0],rect[0,1]:rect[1,1]].mean()

            if mean_int > min_int:
                masks_filtered.append(item)

    return masks_filtered

def do_boxes_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    
    Args:
        bbox1 (tuple): The first bounding box (x, y, w, h).
        bbox2 (tuple): The second bounding box (x, y, w, h).
    Returns:
        bool: True if the bounding boxes overlap, False otherwise.
    """
    x1_min, y1_min, w1, h1 = bbox1
    x2_min, y2_min, w2, h2 = bbox2
    
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    # Check for no overlap
    if x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min:
        return False
    return True

def remove_overlapping_objects(objects):
    """
    Remove objects with overlapping bounding boxes, keeping the one with the higher 'predicted_iou'.
    
    Args:
        objects (list): The list of objects to filter.
    Returns:
        list: The filtered list of objects
    """
    objects = sorted(objects, key=lambda x: x['predicted_iou'], reverse=True)  # Sort by iou descending
    remaining_objects = []

    while objects:
        current = objects.pop(0)
        remaining_objects.append(current)
        # Remove objects that overlap with the current one
        objects = [
            obj for obj in objects
            if not do_boxes_overlap(current['bbox'], obj['bbox'])
        ]

    return remaining_objects

def xywh_to_corners(xywh, frame = 4):
    '''
    Function to convert xywh to corners.
    
    Args:
        xywh (np.array): The xywh coordinates.
        frame (int): The frame to add to the corners.
    Returns:
        np.array: The corners of the bounding box.
    '''
    x, y, w, h = xywh
    return np.array([
        [y-frame, x-frame],          # Top-left (swap x and y)
        [y-frame, x + w + frame],      # Top-right (swap x and y)
        [y + h + frame, x + w + frame],  # Bottom-right (swap x and y)
        [y + h + frame, x - frame],      # Bottom-left (swap x and y)
    ])

def get_refined_rectangles(masks, im, frame, min_area, max_area, min_iou, min_stability, min_int):
    '''
    An envelope function to get refined rectangles.

    It selects the objects based on area, iou, stability and intensity 
    and then removes the overlapping objects prioritizing objects with higher iou.
    
    Finally objects are translated to the rectangles (corners).
    '''
    
    masks_pre_selected = pre_select_objects(masks, im, min_area, max_area, min_iou, min_stability, min_int)
    
    masks_non_overlapping = remove_overlapping_objects(masks_pre_selected)

    rect_list = list(map(lambda mask: xywh_to_corners(mask['bbox'], frame=frame), masks_non_overlapping))
    
    return rect_list

def create_bbox(polygon):
    '''
    Function to create a bounding box from a polygon.
    
    Args:
        polygon: list of points in the polygon
    Returns:
        bbox: bounding box for the polygon
    '''
    
    x_min = int(np.min(polygon[:,0]))
    x_max = int(np.max(polygon[:,0]))
    y_min = int(np.min(polygon[:,1]))
    y_max = int(np.max(polygon[:,1]))
    
    return ([x_min, x_max, y_min, y_max])

def prepare_polygons(poly_data, req_level, org_im_shape):
    '''
    Function to prepare the polygons for saving.
    Polygons are rescaled to the original image size and trimmed to the image frame.

    Args:
        poly_data: list of polygons
        req_level: resolution level used for the polygon definition
        org_im_shape: shape of the original image

    Returns:
        trim_poly_list: list of rescaled and trimmed polygons
    '''

    # get vertices for the original image
    frame_vertices = xywh_to_corners([0,0,org_im_shape[1],org_im_shape[0]], frame = 0)
    frame = Polygon(frame_vertices)

    trim_poly_list = []
    for polygon_vertices in poly_data:

        # rescale the polygons to the original image size
        polygon_vertices = np.array(polygon_vertices).astype(int) * (2**req_level)

        polygon = Polygon(polygon_vertices)

        # Compute the intersection
        intersection = polygon.intersection(frame)

        # add the polygon to the list
        if not intersection.is_empty:
            trim_poly_list.append(np.array(intersection.exterior.coords)[:-1,:])
        else:
            trim_poly_list.append(None)

    return trim_poly_list

def sort_cores(df):
    '''
    Function to sort the cores in the dataframe 
    based on the row and column start values.
    
    Args:
        df: dataframe containing the cores
    Returns:
        df: dataframe with the sorted cores
    '''

    # Round the columns and sort
    df = df.assign(
        row_rounded=lambda x: np.around(x['row_start']/2, decimals=-3),
        col_rounded=lambda x: np.around(x['column_start']/2, decimals=-3),
    ).sort_values(by=['row_rounded', 'col_rounded']).reset_index(drop=True)

    # Drop the temporary columns if not needed
    df = df.drop(columns=['row_rounded', 'col_rounded'])
    
    return df

def prepare_poly_df_for_saving(poly_data, poly_types, req_level, org_im_shape):
    '''
    Function to prepare the dataframe with polygons for saving.

    Args:
        poly_data: list of polygons
        poly_types: list of polygon types
        req_level: resolution level used for the polygon definition
        org_im_shape: shape of the original image

    Returns:
        df: dataframe with the polygons
    '''

    # initialize the dataframe
    df = pd.DataFrame(columns = ['core_name','row_start','row_stop','column_start','column_stop','poly_type','polygon_vertices']).astype({
        'core_name': 'str',
        'row_start': 'int',
        'row_stop': 'int',
        'column_start': 'int',
        'column_stop': 'int',
        'poly_type': 'str',
        'polygon_vertices': 'object' 
    })

    # populate polygon info
    df['poly_type'] = poly_types

    # populate polygon vertices
    trim_poly_list = prepare_polygons(poly_data, req_level, org_im_shape)
    df['polygon_vertices'] = trim_poly_list

    # drop None polygons
    df = df.dropna(subset=['polygon_vertices'])

    # create the array of bounding boxes
    bbox_array = np.array(list(map(lambda poly: create_bbox(poly), df.polygon_vertices.to_list())))
    df.loc[:,['row_start','row_stop','column_start','column_stop']] = bbox_array

    # arrange polygons
    df = sort_cores(df)

    # add core names
    df['core_name'] = [f'Core_{str(i).zfill(3)}' for i in range(len(df))]

    return df

def get_visual_rectangles(df, req_level):
    '''
    Function to get bounderies of the polygons from the dataframe and adjust them to the requested resolution level.

    Args:
        df: dataframe with the polygons
        req_level: requested resolution level
    Returns:
        rect_list: list of rectangles
    '''

    rect_list = df.apply(lambda x: xywh_to_corners([x['column_start'], x['row_start'], x['column_stop'] - x['column_start'], x['row_stop'] - x['row_start']]), axis = 1).tolist()
    rect_list = [(x/(2**(req_level))).astype('int') for x in rect_list]

    return rect_list

def read_in_saved_rois(save_path, IM_LEVEL):
    '''
    Read in the saved rois from the given path.
    '''
    if os.path.exists(save_path.replace('.csv','.pkl')):
        df = pkl.load(open(save_path.replace('.csv','.pkl'),'rb'))
        rect_list = get_visual_rectangles(df, IM_LEVEL)
        poly_list = [(x/(2**IM_LEVEL)).astype('int') for x in df.polygon_vertices.to_list()]
        return rect_list, poly_list, df
    else:
        return [], [], None