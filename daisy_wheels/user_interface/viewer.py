import traceback
from pathlib import Path

import numpy as np
import panel as pn
import napari

from matipo.util.fft import fft_reconstruction

from .general import general_functions
from .panes import panes


class viewer_buttons:
    
    def __init__(self, general_functions, panes):
        """
        Initialise parameters to be updated with button clicks.
        """
        self.g = general_functions
        self.panes = panes
        self.viewer = None
        self.kdata = None
        self.scan_info = None


    def wire(self, widget_dict, save_dict, file):
        self.widget_dict = widget_dict
        self.save_dict = save_dict
        self.file = file # where the UIfunctions is kept
        
    @staticmethod
    def freq_FOV(g_cal, g_val, t_dw):
        """
        g_cal: Hz/m
        g_val: T/m
        t_dw: secs
        """
        return 1/(g_cal * g_val * t_dw) # m
    
    @staticmethod
    
    def phase_FOV(g_cal, g_val, n_phase, t_phase, t_ramp):
        A = g_val * (t_phase + t_ramp)
        delta_k = g_cal * (2 * A / n_phase)
        return 1 / delta_k

    def scale_voxels(self, scan_parameters, file = None):
        """
        calculates the FOV from the scan parameters for 3D scans to scale voxels in the image to mm
        """
        
        t_preread = self.get_t_preread(scan_parameters)
        scale = {"g_read": None, "g_phase1": None, "g_phase2": None}
        axis_g = {"g_read": None, "g_phase1": None, "g_phase2": None}
        for k, v in scan_parameters['img_settings'].items():
            if k == "g_read":
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)

                freq_F = self.freq_FOV(g_cal = g_strength, g_val = v["g"], t_dw = scan_parameters["t_dw"])
                scale[k] = freq_F / v["n"]
                axis_g[k] = v["axis"]
            
            elif k in ["g_phase1", "g_phase2"]:
            
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)
                phase_F = self.phase_FOV(g_cal = g_strength, g_val = v["g"], n_phase = v["n"], t_phase = scan_parameters['t_phase'], t_ramp = scan_parameters['t_ramp'])
                scale[k] = phase_F / v["n"]
                axis_g[k] = v["axis"]
        self.scale_voxel = (scale["g_phase2"]*1000, scale["g_phase1"]*1000, scale["g_read"]*1000) # in mm
        self.axes_g = (axis_g["g_phase2"], axis_g["g_phase1"], axis_g["g_read"])
        self.g.log_msg(f"Voxel size ({self.axes_g}): {self.scale_voxel} mm")


    def scale_voxels2D(self, scan_parameters, file = None):
        """
        calculates the FOV from the scan parameters for 2D scans to scale the pixels to mm
        """
        
        self.g.log_msg(f"{scan_parameters}")
        scale = {"g_read": None, "g_phase": None}
        axis_g = {"g_read": None, "g_phase": None}
        for k, v in scan_parameters["img_settings"].items():
            if k == "g_read":
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)

                freq_F = self.freq_FOV(g_cal = g_strength, g_val = v["g"], t_dw = scan_parameters["t_dw"])
                scale[k] = freq_F / v["n"]
                axis_g[k] = v["axis"]
            
            elif k in ["g_phase"]:
            
                g_strength = self.g.get_g_strength(v["g"], v['axis'], file = file)
                phase_F = self.phase_FOV(g_cal = g_strength, g_val = v["g"], n_phase = v["n"], t_phase = scan_parameters['t_phase'], t_ramp = scan_parameters['t_ramp'])
                scale[k] = phase_F / v["n"]
                axis_g[k] = v["axis"]
                
        self.scale_voxel = (scale["g_phase"]*1000, scale["g_read"]*1000) # in mm
        self.axes_g = (axis_g["g_phase"], axis_g["g_read"])
        self.g.log_msg(f"Voxel size ({self.axes_g}): {self.scale_voxel} mm")

    def get_file(self):
        """
        finds the file requested and loads the scan information and kdata into global variables
        """
        if "fileheader_rare" in self.save_dict.keys():
            file_title = self.save_dict["import_kdata_rare"].value
            fileheader = Path(self.save_dict["fileheader_rare"].value)
        elif "fileheader_2d" in self.save_dict.keys():
            file_title = self.save_dict["import_kdata_2d"].value
            fileheader = Path(self.save_dict["fileheader_2d"].value)
        else:
            file_title = self.save_dict["import_kdata"].value
            fileheader = Path(self.save_dict["fileheader"].value)
        filename = fileheader / file_title
        self.file_title = file_title

        
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")
            
        
        try:
            data = np.load(filename, allow_pickle=True)
            self.g.log_msg(f"File found: {filename}")
        except Exception:
            self.g.log_msg(f"!!! Error !!! No data_file was given or filepath does not exist. Filepath tried: {filename}")
            return

        kdata = data["kdata"] # will be complex, 1D array
        self.kdata = kdata
        scan_info = data["scan_info"].item()
        self.g.log_msg(f"Loaded Scan Information: {scan_info}")

        return kdata, scan_info
        
    def img_run_on_click(self, event):
        """
        opens a scaled napari image depending on the scan_type and the number of dimensions to get the kdata and image in napari. 
        The viewer will open with all images hidden so click the eye icon on the image label to view it 
        """
        try:
            
            kdata, scan_info = self.get_file()
        except:
            ValueError("Doesn't exist")
            return
        self.scan_info = scan_info
                
        g_blur = self.widget_dict["gaussian_blur"].value
    
        if scan_info['dimensions'] == 1:
            self.g.log_msg(log_pane, f"!!! Error !!! \nThis method is not suitable for 1D scans.")
            return
        
        if scan_info['dimensions'] == 2:
            scan_type = scan_info["scan_type"]
            try:
                self.viewer2 = napari.Viewer(title = self.file_title, ndisplay=2)

                for var, data in enumerate(kdata): 

                    try:
                        if scan_type in ["rare", "rare_pe"]:
                            data = data[:int(scan_info['n_phase']*scan_info['n_samples'])]
                          
                        kdata1 = np.reshape(data, (scan_info['n_phase'], scan_info['n_samples']))
                    except Exception:
                        tb = traceback.format_exc()
                        self.g.log_msg("Error in data transformation:")
                        self.g.log_msg(tb)
                        return
                    self.kdata = kdata1
                    self.scale_voxels2D(scan_info, file = self.file)

                    self.viewer2.dims.axis_labels = self.axes_g
    
                        
                    self.g.log_msg(f"kdata shape before open napari: {np.shape(self.kdata)}")
        
                    if scan_type in ["2DSE_multi", 'rare', "rare_pe", '2DSE_multiSLR']:
                        k_layer = self.viewer2.add_image(np.abs(self.kdata), name=f"Kdata {var} Slice: {scan_info["slices_list"][var]}mm", scale = (1,1))
                    else:
                        k_layer = self.viewer2.add_image(np.abs(self.kdata), name=f"k_space", scale = (1,1))

                    k_layer.visible = False
                    k_layer.scale = (1, 1)
                    self.g.log_msg(f"Viewer Opened with Gaussian Values Selected: {g_blur}")
                    for i in g_blur:
                        image = fft_reconstruction(self.kdata, gaussian_blur=i, upscale_factor=1)
                        self.g.log_msg(f"image shape: {np.shape(image)}")
                        if scan_type in ["2DSE_multi", 'rare', "rare_pe", '2DSE_multiSLR']:
                            img_layer = self.viewer2.add_image(np.abs(image), name=f"Image {var} Slice: {scan_info["slices_list"][var]}mm")
                        else:    
                            img_layer = self.viewer2.add_image(np.abs(image), name=f"Img_guassian{i}")
                        img_layer.units = ("mm", "mm")
                        img_layer.visible = False
            
                        # Make image layer active so scale bar shows mm
                        self.viewer2.layers.selection.clear()
                        self.viewer2.layers.selection.add(img_layer)
            
                    
                    self.viewer2.scale_bar.visible = True
                    self.viewer2.scale_bar.unit = "mm"
                    k_layer.gamma = 1
                    self.viewer2.axes.visible = True
                    k_layer.interpolation2d = 'nearest'
                    k_layer.interpolation3d = 'nearest'
                    k_layer.rendering = 'mip'
                    k_layer.projection_mode = 'none'
        
                    img_layer.gamma = 1
                    img_layer.interpolation2d = 'nearest'
                    img_layer.interpolation3d = 'nearest'
                    img_layer.rendering = 'mip'
                    img_layer.projection_mode = 'none'
                    
    
            except Exception:
                tb = traceback.format_exc()
                self.g.log_msg("Error in opening viewer:")
                self.g.log_msg(tb)
                return                

            
            return
            
        if scan_info['dimensions'] == 3:
            self.viewer = napari.Viewer(title = self.file_title, ndisplay=3)
            try:
                kdata = np.reshape(kdata, (scan_info['n_phase2'], scan_info['n_phase1'], scan_info['n_samples'])) # reshape the data to the dimensions
                self.scale_voxels(scan_info, file = self.file)
                self.viewer.dims.axis_labels = self.axes_g
                k_layer = self.viewer.add_image(np.abs(kdata), name=f"k_space ",  units = "", scale = (1,1,1))
                k_layer.scale = (1, 1, 1)
                k_layer.projection_mode = 'none'
                self.g.log_msg(f"Run with Gaussian Values Selected: {g_blur}")

                self.viewer.scale_bar.visible = True
                self.viewer.scale_bar.unit = "mm"
                k_layer.gamma = 1
                self.viewer.axes.visible = True
                k_layer.interpolation2d = 'nearest'
                k_layer.interpolation3d = 'nearest'
                k_layer.rendering = 'mip' 
                k_layer.visible = False
        
                for i in g_blur:
                    image = fft_reconstruction(kdata, gaussian_blur=i, upscale_factor=2)
                    img_layer = self.viewer.add_image(np.abs(image), name=f"Img_guassian")
                    img_layer.projection_mode = 'none'
                    img_layer.scale = self.scale_voxel
                    img_layer.units = ("mm", "mm", "mm")
                    img_layer.visible = False
                    

                    img_layer.gamma = 1
                    img_layer.interpolation2d = 'nearest'
                    img_layer.interpolation3d = 'nearest'
                    img_layer.rendering = 'mip'

                    
            except Exception:
                tb = traceback.format_exc()
                self.g.log_msg("Error in data transformation:")
                self.g.log_msg(tb)
                return


    def img_add_on_click(self, event):
        """
        Adds the image and kdata requested into the open napari browser 
        """
        try:
            
            kdata, scan_info = self.get_file()
            scan_type = scan_info["scan_type"]
        except:
            ValueError("Doesn't exist")
            return
        self.scan_info = scan_info
                
        g_blur = self.widget_dict["gaussian_blur"].value
    
        if scan_info['dimensions'] == 1:
            self.g.log_msg(log_pane, f"!!! Error !!! \nThis method is not suitable for 1D scans.")
            return
        
        if scan_info['dimensions'] == 2:
               for var, data in enumerate(kdata): 

                    try:
                        if scan_type in ["rare", "rare_pe"]:
                            data = data[:int(scan_info['n_phase']*scan_info['n_samples'])]
                          
                        kdata1 = np.reshape(data, (scan_info['n_phase'], scan_info['n_samples']))
                    except Exception:
                        tb = traceback.format_exc()
                        self.g.log_msg("Error in data transformation:")
                        self.g.log_msg(tb)
                        return
                    self.kdata = kdata1
                    self.scale_voxels2D(scan_info, file = self.file)

                    self.viewer2.dims.axis_labels = self.axes_g
    
                        
                    self.g.log_msg(f"kdata shape before open napari: {np.shape(self.kdata)}")
        
                    if scan_type in ["2DSE_multi", 'rare', "rare_pe", '2DSE_multiSLR']:
                        k_layer = self.viewer2.add_image(np.abs(self.kdata), name=f"Kdata {var} Slice: {scan_info["slices_list"][var]}mm", scale = (1,1))
                    else:
                        k_layer = self.viewer2.add_image(np.abs(self.kdata), name=f"k_space", scale = (1,1))

                    k_layer.visible = False
                    k_layer.scale = (1, 1)
                    self.g.log_msg(f"Viewer Opened with Gaussian Values Selected: {g_blur}")
                    for i in g_blur:
                        image = fft_reconstruction(self.kdata, gaussian_blur=i, upscale_factor=1)
                        self.g.log_msg(f"image shape: {np.shape(image)}")
                        if scan_type in ["2DSE_multi", 'rare', "rare_pe", '2DSE_multiSLR']:
                            img_layer = self.viewer2.add_image(np.abs(image), name=f"Image {var} Slice: {scan_info["slices_list"][var]}mm")
                        else:    
                            img_layer = self.viewer2.add_image(np.abs(image), name=f"Img_guassian{i}")
                        img_layer.units = ("mm", "mm")
                        img_layer.visible = False
            
                        # Make image layer active so scale bar shows mm
                        self.viewer2.layers.selection.clear()
                        self.viewer2.layers.selection.add(img_layer)
            
                    
                    self.viewer2.scale_bar.visible = True
                    self.viewer2.scale_bar.unit = "mm"
                    k_layer.gamma = 1
                    self.viewer2.axes.visible = True
                    k_layer.interpolation2d = 'nearest'
                    k_layer.interpolation3d = 'nearest'
                    k_layer.rendering = 'mip'
                    k_layer.projection_mode = 'none'
        
                    img_layer.gamma = 1
                    img_layer.interpolation2d = 'nearest'
                    img_layer.interpolation3d = 'nearest'
                    img_layer.rendering = 'mip'
                    img_layer.projection_mode = 'none'
                    
            
        if scan_info['dimensions'] == 3:

            try:
                kdata = np.reshape(kdata, (scan_info['n_phase2'], scan_info['n_phase1'], scan_info['n_samples'])) # reshape the data to the dimensions
                self.scale_voxels(scan_info, file = self.file)
                self.viewer.dims.axis_labels = self.axes_g
                k_layer = self.viewer.add_image(np.abs(kdata), name=f"k_space {self.widget_dict["z_factor"].value, self.widget_dict["y_factor"].value, self.widget_dict["x_factor"].value}",  units = "", scale = (1,1,1))
                k_layer.scale = (1, 1, 1)
                k_layer.projection_mode = 'none'
                self.g.log_msg(f"Run with Gaussian Values Selected: {g_blur}")

                self.viewer.scale_bar.visible = True
                self.viewer.scale_bar.unit = "mm"
                k_layer.gamma = 1
                self.viewer.axes.visible = True
                k_layer.interpolation2d = 'nearest'
                k_layer.interpolation3d = 'nearest'
                k_layer.rendering = 'mip'
                k_layer.visible = False

        
                for i in g_blur:
                    image = fft_reconstruction(kdata, gaussian_blur=i, upscale_factor=2)
                    img_layer = self.viewer.add_image(np.abs(image), name=f"Img_guassian{self.widget_dict["z_factor"].value, self.widget_dict["y_factor"].value, self.widget_dict["x_factor"].value}")
                    img_layer.projection_mode = 'none'
                    img_layer.scale = self.scale_voxel
                    img_layer.units = ("mm", "mm", "mm")
                    img_layer.visible = False


                    img_layer.gamma = 1
                    img_layer.interpolation2d = 'nearest'
                    img_layer.interpolation3d = 'nearest'
                    img_layer.rendering = 'mip'
            
                    
            except Exception:
                tb = traceback.format_exc()
                self.g.log_msg("Error in data transformation:")
                self.g.log_msg(tb)
                return