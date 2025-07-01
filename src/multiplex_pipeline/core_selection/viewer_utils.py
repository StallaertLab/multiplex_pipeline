"""Napari viewer helpers for displaying and saving ROI layers."""

from qtpy.QtWidgets import QFileDialog

from multiplex_pipeline.roi_utils import read_in_saved_rois, prepare_poly_df_for_saving, get_visual_rectangles


def redo_cores_layer(viewer, data=[], shape_type="polygon"):
    """Create or replace the ``cores`` layer in a viewer.

    Args:
        viewer (napari.Viewer): Viewer instance to update.
        data (list, optional): Vertices describing the shapes. Defaults to ``[]``.
        shape_type (str, optional): Napari shape type. Defaults to ``"polygon"``.

    Returns:
        None
    """

    if "cores" in viewer.layers:
        viewer.layers.remove("cores")

    viewer.add_shapes(
        data,
        shape_type=shape_type,
        edge_color="green",
        face_color="transparent",
        edge_width=2,
        name="cores",
    )
    
def redo_bbox_layer(viewer, data=[], text=[]):
    """Create or replace the ``bounding_boxes`` layer in a viewer.

    Args:
        viewer (napari.Viewer): Viewer instance to update.
        data (list, optional): Rectangle coordinates. Defaults to ``[]``.
        text (list, optional): Labels shown next to rectangles. Defaults to ``[]``.

    Returns:
        None
    """

    if "bounding_boxes" in viewer.layers:
        viewer.layers.remove("bounding_boxes")

    viewer.add_shapes(
        data,
        shape_type="rectangle",
        edge_color="red",
        face_color="transparent",
        edge_width=2,
        name="bounding_boxes",
        text={"string": text, "size": 12, "color": "red", "anchor": "upper_left"},
    )

def display_saved_rois(viewer, IM_LEVEL, save_path=None):
    """Load ROI annotations from disk and show them in the viewer.

    Args:
        viewer (napari.Viewer): Viewer instance where layers are added.
        IM_LEVEL (int): Image pyramid level used when the ROIs were saved.
        save_path (str, optional): File path of the saved ROIs. Defaults to ``None``.

    Returns:
        None
    """
    rect_list, poly_list, df = read_in_saved_rois(save_path, IM_LEVEL = IM_LEVEL)
    if len(rect_list) > 0:
        redo_bbox_layer(viewer,rect_list,df['core_name'].tolist())
        redo_cores_layer(viewer,poly_list,shape_type = df.poly_type.to_list())
    else:
        viewer.status = 'No previous rois found!'

def save_rois_from_viewer(viewer, org_im_shape, req_level, save_path=None):
    """Save ROIs drawn in the viewer to disk and update the displayed layers.

    Args:
        viewer (napari.Viewer): Viewer containing a ``cores`` shapes layer.
        org_im_shape (tuple[int, int]): Shape of the original image.
        req_level (int): Resolution level at which the ROIs are defined.
        save_path (str, optional): Destination CSV file path. Defaults to ``None``.

    Returns:
        None
    """
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