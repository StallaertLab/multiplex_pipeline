from qtpy.QtWidgets import QFileDialog

from multiplex_pipeline.roi_utils import read_in_saved_rois, prepare_poly_df_for_saving, get_visual_rectangles


def redo_cores_layer(viewer,data=[],shape_type='polygon'):

    if 'cores' in viewer.layers:
        viewer.layers.remove('cores')

    viewer.add_shapes(
    data,       
    shape_type=shape_type,  
    edge_color='green', 
    face_color='transparent',  
    edge_width=2,       
    name = 'cores'
)
    
def redo_bbox_layer(viewer,data=[],text=[]):

    if 'bounding_boxes' in viewer.layers:
        viewer.layers.remove('bounding_boxes')

    viewer.add_shapes(
    data,       
    shape_type='rectangle',  
    edge_color='red', 
    face_color='transparent',  
    edge_width=2,       
    name = 'bounding_boxes',
    text = {'string': text,'size':12,'color':'red','anchor':'upper_left'}
) 

def display_saved_rois(viewer, IM_LEVEL, save_path = None):
    '''
    Display the saved rois from the file in the viewer.
    '''
    rect_list, poly_list, df = read_in_saved_rois(save_path, IM_LEVEL = IM_LEVEL)
    if len(rect_list) > 0:
        redo_bbox_layer(viewer,rect_list,df['core_name'].tolist())
        redo_cores_layer(viewer,poly_list,shape_type = df.poly_type.to_list())
    else:
        viewer.status = 'No previous rois found!'

def save_rois_from_viewer(viewer, org_im_shape, req_level, save_path = None):
    '''
    '''
    if 'cores' in viewer.layers:

        # get the saving path if not provided
        if save_path is None:
            # open dialog for getting a dir to save csv file
            save_path = QFileDialog.getSaveFileName(filter = 'CSV file (*.csv)')[0]

        # get the polygon data
        poly_data = viewer.layers['cores'].data
        poly_types = viewer.layers['cores'].shape_type

        # prepare df for saving
        df = prepare_poly_df_for_saving(poly_data, poly_types, req_level, org_im_shape)

        # save the rois
        df.to_pickle(save_path.replace('.csv','.pkl'))
        df.to_csv(save_path, index = False)

        # prepare the cores visual for saving
        rect_list = get_visual_rectangles(df, req_level)
        poly_list = [(x/(2**(req_level))).astype('int') for x in df.polygon_vertices.to_list()]

        # change the visualization
        redo_cores_layer(viewer,poly_list,shape_type = df.poly_type.to_list())
        redo_bbox_layer(viewer,rect_list,df['core_name'].tolist())

        # get a screenshot of the viewer
        screenshot_path = save_path.replace('.csv', '_screenshot.png')
        # reset zoom
        viewer.reset_view()
        viewer.screenshot(screenshot_path, canvas_only=True)

        viewer.status = f'Cores saved to {save_path}'

    else:
        viewer.status = 'No layer called "cores" found!'