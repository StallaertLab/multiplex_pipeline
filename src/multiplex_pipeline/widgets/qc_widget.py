from functools import partial

from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import numpy as np
from geopandas import GeoDataFrame
from shapely import Polygon
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

class QCWidget(QWidget):

    def __init__(self, napari_viewer: "Viewer", sdata) -> None:

        super().__init__()
        self.setLayout(QVBoxLayout())

        self.position = 0
        self.viewer = napari_viewer
        self.sdata = sdata
        self.im_list = sorted(list(sdata.images.keys()))  # stable order
        self.len = len(self.im_list)
        self.im_name = self.im_list[self.position]
        self.shapes_name = f'qc_exclude_{self.im_name}'

        navigation_group = QGroupBox()
        navigation_group.setLayout(QVBoxLayout())

        navigation_group.layout().addWidget(QLabel("QC:"))
        self.navigation_row = self.add_navigation_control()
        navigation_group.layout().addWidget(self.navigation_row)
        self.position_label = QLabel()
        navigation_group.layout().addWidget(self.position_label)
        
        self.shapes_row = self.add_shapes_row()
        navigation_group.layout().addWidget(self.shapes_row)

        self.layout().addWidget(navigation_group)

        save_btn = self.add_save_btn()
        self.layout().addWidget(save_btn)


        if self.len:
            self.show_current()
        self.update_position_label()

    def update_position_label(self):
        if self.len:
            self.position_label.setText(f"Position: {self.position}/{self.len - 1}")
        else:
            self.position_label.setText("No images loaded")

    def add_navigation_control(self) -> QWidget:

        navigation_row = QWidget()
        navigation_row.setLayout(QGridLayout())

        self.backward_btn = self.add_backward_btn()
        self.forward_btn = self.add_forward_btn()

        navigation_row.layout().addWidget(self.backward_btn, 0, 0)
        navigation_row.layout().addWidget(self.forward_btn, 0, 1)

        return navigation_row
    
    def add_backward_btn(self) -> QPushButton:

        backward_btn = QPushButton("<")

        backward_btn.clicked.connect(partial(self.step, True))

        return backward_btn
    
    def add_forward_btn(self) -> QPushButton:

        forward_btn = QPushButton(">")

        forward_btn.clicked.connect(partial(self.step, False))

        return forward_btn
    
    def clear_viewer(self):
        for layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)

    def show_current(self):
        # load new data
        self.viewer.add_image(
            [self.sdata.images[self.im_name][k].image for k in self.sdata.images[self.im_name].keys()],
            visible=True,
            name = self.im_name,
            blending = 'additive',
        )

        if self.shapes_name in self.sdata:
            data = [np.array(self.sdata[self.shapes_name].geometry[i].exterior.coords) for i in range(len(self.sdata[self.shapes_name]))]
            self.shapes_layer = self.viewer.add_shapes(data = data, name = self.shapes_name, shape_type = 'polygon')

    def step(self, backward=False):

        # save shapes
        self.remember_shapes()

        if backward:
            if self.position == 0:         
                return 
            else:
                self.position -= 1
        else:
            if self.position == self.len - 1:
                return
            else:
                self.position += 1


        # updata current names
        self.im_name = self.im_list[self.position]
        self.shapes_name = f'qc_exclude_{self.im_name}'

        # clear viewer
        self.clear_viewer()

        # load new data
        self.show_current()

        # update label
        self.update_position_label()

    def numpy_to_shapely(self, x: np.array) -> Polygon:
        return Polygon(list(map(tuple, x)))

    def remember_shapes(self):
        if self.shapes_name in [x.name for x in self.viewer.layers]:
            
            if self.viewer.layers[self.shapes_name].data:
                gdf = GeoDataFrame({"geometry": [self.numpy_to_shapely(x) for x in self.viewer.layers[self.shapes_name].data]})
                gdf = ShapesModel.parse(gdf, transformations={"global": Identity()})
                self.sdata.shapes[self.shapes_name] = gdf
            else:
                if self.shapes_name in self.sdata:
                    del self.sdata[self.shapes_name]
    
    def add_shapes_row(self) -> QWidget:

        shapes_row = QWidget()
        shapes_row.setLayout(QGridLayout())

        self.shapes_btn = self.add_shapes_btn()

        shapes_row.layout().addWidget(self.shapes_btn, 0, 0)

        return shapes_row
    
    def add_shapes_btn(self) -> QPushButton:

        shapes_btn = QPushButton("Add Shapes")

        shapes_btn.clicked.connect(self.add_shapes)

        return shapes_btn
    
    def add_shapes(self):
        
        if self.shapes_name in [x.name for x in self.viewer.layers]:
            self.viewer.layers.remove(self.shapes_name)
            del self.sdata[self.shapes_name]

        self.shapes_layer = self.viewer.add_shapes(
            data=[],                 # start empty
            shape_type='polygon',
            name=self.shapes_name,
            #edge_width=2,
            #edge_color='yellow',
            #face_color=[0, 0, 0, 0],  # transparent fill
        )

        self.viewer.layers.selection.active = self.shapes_layer

    def add_save_btn(self) -> QPushButton:

        save_btn = QPushButton("Save")

        save_btn.clicked.connect(self.save_shapes_layer)

        return save_btn

    def save_shapes_layer(self):
        self.sdata.write_element(self.shapes_name)
        self.viewer.status = f'{self.shapes_name} has been saved to disk.'

    def save_shapes_all(self):
        pass

    def create_global_mask(self):
        pass