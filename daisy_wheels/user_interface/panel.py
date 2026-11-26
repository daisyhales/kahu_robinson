from matipo import SEQUENCE_DIR, GLOBALS_DIR
from matipo.sequence import Sequence
import sys

# sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import panel as pn
import yaml
from pathlib import Path
import logging
import traceback

from .general import general_functions
from .panes import panes


class panel_functions:
    """
    Class for the creations of the panel elements that can be used with any scan.
    """
    def __init__(self, input_width = 150, file = None):
        """
        Parameters that are created on initialisation to be used throughout the class.

        input_dict (dict): dictionary containing all the widgets that can be made for the panels with 
            the information that is needed to create the type of widget required. Update with more widgets 
            as required when building new pulse sequences.
            Parameters used:
                widget_type (str): sets the type of widget to be defined - used in filtering to create
                    in the create_parameter_widgets function
                name (str): name to display above the widget
                placeholder (str): used for text input as the text that exists inside the input box before
                    the user writes in it
                sizing_mode (str): how the input will fill the pane it exists in
                format (str): how many decimals to display the slider value as above the slider
                value (int/float): value that starts in the input on initialisation
                start (int/float): min limit to the parameter 
                end (int/float): max limit to the parameter
                step (int/float): incremement steps to the arrows or slider when adjusted
                factor (int/float): if the input is made in a factor of the SI units this will 
                    tell what factor it needs to be multiplied by for the calculations.
            
        button_dict (dict): dictionary of the button widgets that can be created and the info needed to 
            make them
            Parameters used:
                name (str): name on button
                button_type (str): tells colour to make the button
            
        """
        self.input_width = input_width
        self.g = general_functions(panes)
        self.shims_limit, self.amps_limit = self.g.shims_limitation(file)
        current_shims = self.g.get_yaml_info(info_name = "shims", file = file)["offsets"]

        self.input_dict = {
                            "filepath": {
                                "widget_type": "TextInput",
                                "name": "K Space Data Location",
                                "value" : None,
                                "placeholder": 'Enter file name...',
                                "sizing_mode": 'stretch_width',
                            },
                            "n_scans": {
                                "widget_type": "IntInput",
                                "name": "Number of Scans",
                                "value": 1,
                                "start": 1,
                                "end": None,
                                "step": 1,
                            },
                            "t_dw": {
                                "widget_type": "FloatInput",
                                "name": "Dwell Time (μs)",
                                "value": 40,
                                "start": None,
                                "end": None,
                                "step": 1,
                                "factor": 1e-6
                            },
                            "n_samples": {
                                "widget_type": "IntInput",
                                "name": "Number of Samples",
                                "value": 32,
                                "start": 1,
                                "end": None,
                                "step": 16,
                            },
                            "t_echo": {
                                "widget_type": "FloatInput",
                                "name": "Echo Time (μs)",
                                "value": 2500,
                                "start": None,
                                "end": None,
                                "step": 1000,
                                "factor": 1e-6
                            },
                            "t_end": {
                                "widget_type": "FloatInput",
                                "name": "T End (s)",
                                "value": 1,
                                "start": None,
                                "end": None,
                                "step": 0.1,
                            },
                            "select_xyz": {
                                "widget_type": "Select",
                                "name": "X, Y or Z Gradient",
                                "options": ["X", "Y", "Z"],
                                "value": "X"
                            },
                            "x": {
                                "widget_type": "FloatInput",
                                "name": "X Gradient (0~0.5)",
                                "value": 0.01,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                            "y": {
                                "widget_type": "FloatInput",
                                "name": "Y Gradient (0~0.3)",
                                "value": 0,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                            "z": {
                                "widget_type": "FloatInput",
                                "name": "Z Gradient (0~0.3)",
                                "value": 0,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                            "t_ramp": {
                                "widget_type": "FloatInput",
                                "name": "Ramp Time (μs)",
                                "value": 250,
                                "start": None,
                                "end": None,
                                "step": 1,
                                "factor": 1e-6
                            },
                            "x_offset": {
                                "widget_type": "FloatSlider",
                                "name": "X Gradient Offset (%)",
                                "value": (current_shims[0] * 100) / self.shims_limit[0],
                                "start": -100,
                                "end": 100,
                                "step": 1,
                            },
                            "y_offset": {
                                "widget_type": "FloatSlider",
                                "name": "Y Gradient Offset (%)",
                                "value": (current_shims[1] * 100) / self.shims_limit[1],
                                "start": -100,
                                "end": 100,
                                "step": 1,
                            },
                            "z_offset": {
                                "widget_type": "FloatSlider",
                                "name": "Z Gradient Offset (%)",
                                "value": (current_shims[2] * 100) / self.shims_limit[2],
                                "start": -100,
                                "end": 100,
                                "step": 1,
                            }, 
                            "n_sweep": {
                                "widget_type": "IntInput",
                                "name": "N Sweep",
                                "value": 1001,
                                "start": 0,
                                "end": None,
                                "step": 10,
                            },
                            "t_acqdelay": {
                                "widget_type": "FloatInput",
                                "name": "Acquisition Delay Time (μs)",
                                "value": 2500,
                                "start": 0,
                                "end": None,
                                "step": 10,
                                "factor":1e-6
                            },
                            "g_crusher": {
                                "widget_type": "FloatInput",
                                "name": "Crusher Gradient",
                                "value": 0,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                         "width_threshold": {
                                "widget_type": "IntInput",
                                "name": "Threshold Width (%)",
                                "value": 1,
                                "start": 0,
                                "end": 100,
                                "step": 1,
                            },
                         "gaussian_blur": {
                                "widget_type": "MultiSelect",
                                "name": "Select Gaussian Blur",
                                "value": [1],
                                "options": [0,1,2,3,4,5,6],
                            },
                         "import_kdata": {
                                "widget_type": "TextInput",
                                "name": "K Space Data Location:",
                                "value": 'array_trial.npz',
                                "placeholder": 'Enter file path...',
                                 "sizing_mode": "stretch_width"
                            },
                         "import_kdata_rare": {
                                "widget_type": "TextInput",
                                "name": "K Space Data Location:",
                                "value": 'n_ETL3.npz',
                                "placeholder": 'Enter file path...',
                                 "sizing_mode": "stretch_width"
                            },   
                          "import_kdata_2d": {
                                "widget_type": "TextInput",
                                "name": "K Space Data Location:",
                                "value": '128_200726_3_lobe.npz',
                                "placeholder": 'Enter file path...',
                                 "sizing_mode": "stretch_width"
                            },  
                        "select_g_read": {
                                "widget_type": "Select",
                                "name": "X, Y or Z Read Gradient",
                                "options": ["X", "Y", "Z"],
                                "value": "X"
                            },     
                        "select_g_phase1": {
                                "widget_type": "Select",
                                "name": "X, Y or Z Phase 1 Gradient",
                                "options": ["X", "Y", "Z"],
                                "value": "Y"
                            },
                        "select_g_phase2": {
                                "widget_type": "Select",
                                "name": "X, Y or Z Phase 2 Gradient",
                                "options": ["X", "Y", "Z"],
                                "value": "Z"
                            },   
                         "g_read": {
                                "widget_type": "FloatInput",
                                "name": "Read Gradient",
                                "value": 0.01,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                         "g_phase1": {
                                "widget_type": "FloatInput",
                                "name": "Phase 1 Gradient",
                                "value": 0.01,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                         "g_phase2": {
                                "widget_type": "FloatInput",
                                "name": "Phase 2 Gradient",
                                "value": 0.01,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                        "n_phase2": {
                                "widget_type": "IntInput",
                                "name": "Number of Phase 2",
                                "value": 32,
                                "start": 16,
                                "end": None,
                                "step": 16,
                            },            
                        "n_phase1": {
                                "widget_type": "IntInput",
                                "name": "Number of Phase 1",
                                "value": 32,
                                "start": 16,
                                "end": None,
                                "step": 16,
                            },            
                         "soft_pulse_type": {
                                "widget_type": "Select",
                                "name": "Soft Pulse Type",
                                "options": ["SLR", "3-lobe sinc", "5-lobe sinc"],
                                "value": "SLR",
                             
                            },
                         "tau": {
                                "widget_type": "FloatInput",
                                "name": "Tau (s)",
                                "value": 0.5,
                                "start": None,
                                "end": None,
                                "step": 0.1,
                            },            
                         "tau_2": {
                                "widget_type": "FloatInput",
                                "name": "Tau 2 (s)",
                                "value": 0.5,
                                "start": None,
                                "end": None,
                                "step": 0.1,
                            },
                         "a_rf": {
                                "widget_type": "FloatInput",
                                "name": "RF Pulse Amplitude",
                                "value": 0.2,
                                "start": 0,
                                "end": 1,
                                "step": 0.1,
                            },       
                         "fileheader_rare": {
                                "widget_type": "TextInput",
                                "name": "File Header:",
                                "value": r'C:\Users\halesda\OneDrive - Victoria University of Wellington - STAFF\Desktop\kahu-workspace\notebooks\daisy\rare\Panel\kdata',
                                "placeholder": 'Enter file header...',
                                 "sizing_mode": "stretch_width"
                            },    
                        "fileheader_2d": {
                                "widget_type": "TextInput",
                                "name": "File Header:",
                                "value": r'C:\Users\halesda\OneDrive - Victoria University of Wellington - STAFF\Desktop\kahu-workspace\notebooks\daisy\spin_echos\_2DSE\Panel\kdata',
                                "placeholder": 'Enter file header...',
                                 "sizing_mode": "stretch_width"
                            },    
                         "fileheader": {
                                "widget_type": "TextInput",
                                "name": "File Header:",
                                "value": r'C:\Users\halesda\OneDrive - Victoria University of Wellington - STAFF\Desktop\kahu-workspace\notebooks\daisy\spin_echos\_2DSE\kdata',
                                "placeholder": 'Enter file header...',
                                 "sizing_mode": "stretch_width"
                            },     
                         "g_slice_ex": {
                                "widget_type": "FloatInput",
                                "name": "Spoiler/Crusher Gradient",
                                "value": 0,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                        "t_phase": {
                                "widget_type": "FloatInput",
                                "name": "Phase Time (μs)",
                                "value": 2000,
                                "start": 0,
                                "end": None,
                                "step": 10,
                                "factor":1e-6
                            },
                         "g_phaseread": {
                                "widget_type": "FloatInput",
                                "name": "Wound Read Gradient",
                                "value": -0.18,
                                "start": None,
                                "end": None,
                                "step": 0.01,
                            },
                        "t_crush": {
                                "widget_type": "FloatInput",
                                "name": "Crusher Gradient Time (ms)",
                                "value": 1,
                                "start": 0,
                                "end": None,
                                "step": 1,
                                "factor":1e-3
                            },
                        "t_spoil": {
                                "widget_type": "FloatInput",
                                "name": "Spoil Gradient Time (ms)",
                                "value": 1,
                                "start": 0,
                                "end": None,
                                "step": 1,
                                "factor":1e-3
                            },
                        "rf_number": {
                                "widget_type": "IntInput",
                                "name": "Number of RF Tried",
                                "value": 50,
                                "start": 0,
                                "end": None,
                                "step": 10,
                            },    
                     "a_rf_slider": {
                            "widget_type": "FloatSlider",
                            "name": "RF Amplitude (%)",
                            "value": 0,
                            "start": 0,
                            "end": 100,
                            "step": 1,
                        },
                     "select_g_phase": {
                            "widget_type": "Select",
                            "name": "X, Y or Z Phase Gradient",
                            "options": ["X", "Y", "Z"],
                            "value": "Y"
                        },
                     "g_phase": {
                            "widget_type": "FloatInput",
                            "name": "Phase Gradient",
                            "value": 0.01,
                            "start": None,
                            "end": None,
                            "step": 0.01,
                        },
                    "n_phase": {
                            "widget_type": "IntInput",
                            "name": "Number of Phases",
                            "value": 32,
                            "start": 0,
                            "end": None,
                            "step": 16,
                        },         
                    "select_g_slice": {
                            "widget_type": "Select",
                            "name": "X, Y or Z Slice Gradient",
                            "options": ["X", "Y", "Z"],
                            "value": "Z"
                        },
                    "g_slice_90": {
                            "widget_type": "FloatInput",
                            "name": "Slice 90 deg Gradient",
                            "value": 0,
                            "start": None,
                            "end": None,
                            "step": 0.01,
                        },
                    "g_slice_180": {
                            "widget_type": "FloatInput",
                            "name": "Slice 180 deg Gradient",
                            "value": 0,
                            "start": None,
                            "end": None,
                            "step": 0.01,
                        },      
                    "x_factor": {
                            "widget_type": "FloatInput",
                            "name": "X",
                            "value": 1,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        },
                    "y_factor": {
                            "widget_type": "FloatInput",
                            "name": "Y",
                            "value": 1,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        },
                    "z_factor": {
                            "widget_type": "FloatInput",
                            "name": "Z",
                            "value": 1,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        },
                    "spectral_width": {
                            "widget_type": "FloatInput",
                            "name": "Spectral Width (kHz)",
                            "value": 50,
                            "start": 0,
                            "end": None,
                            "step": 5,
                        },            
                    "FOV_read": {
                            "widget_type": "FloatInput",
                            "name": "FOV read (mm)",
                            "value": 160,
                            "start": None,
                            "end": None,
                            "step": 5,
                        },            
                    "resolution": {
                            "widget_type": "FloatInput",
                            "name": "Resolution (pixels)",
                            "value": 32,
                            "start": 0,
                            "end": None,
                            "step": 16,
                        },       
           
                    "resolution_read": {
                            "widget_type": "FloatInput",
                            "name": "Resolution (pixels)",
                            "value": 32,
                            "start": 0,
                            "end": None,
                            "step": 16,
                        },               
                    "FOV_phase1": {
                            "widget_type": "FloatInput",
                            "name": "FOV Phase 1 (mm)",
                            "value": 160,
                            "start": None,
                            "end": None,
                            "step": 5,
                        },            
                    "resolution_phase1": {
                            "widget_type": "FloatInput",
                            "name": "Resolution (pixels)",
                            "value": 32,
                            "start": 0,
                            "end": None,
                            "step": 16,
                        },  
                    "FOV_phase2": {
                            "widget_type": "FloatInput",
                            "name": "FOV Phase 2 (mm)",
                            "value": 160,
                            "start": None,
                            "end": None,
                            "step": 5,
                        },            
                    "resolution_phase2": {
                            "widget_type": "FloatInput",
                            "name": "Resolution (pixels)",
                            "value": 32,
                            "start": 0,
                            "end": None,
                            "step": 16,
                        },  
                    "slice_thickness": {
                            "widget_type": "FloatInput",
                            "name": "Slice Thickness (mm)",
                            "value": 5,
                            "start": None,
                            "end": None,
                            "step": 1,
                        },         
                    "echo_train_length": {
                            "widget_type": "IntInput",
                            "name": "Echo Train Length",
                            "value": 2,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        }, 
                    "te_eff": {
                            "widget_type": "FloatInput",
                            "name": "TE Effective (ms)",
                            "value": 60,
                            "start": 0,
                            "end": None,
                            "step": 5,
                        }, 
                    "t_rep": {
                            "widget_type": "FloatInput",
                            "name": "Rep Time (s)",
                            "value": 1,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        },  
                    "slice_offset": {
                            "widget_type": "FloatInput",
                            "name": "Slice Offset (mm)",
                            "value": 0,
                            "start": None,
                            "end": None,
                            "step": 1,
                        },   
                    "empty_tr": {
                            "widget_type": "FloatInput",
                            "name": "TR (s)",
                            "value": 0,
                            "start": 0,
                            "end": None,
                            "step": 1,
                            "disabled": True
                        },  

                     "TR": {
                            "widget_type": "FloatInput",
                            "name": "TR (s)",
                            "value": 1,
                            "start": 0,
                            "end": None,
                            "step": 0.1,
                        }, 
                     "slice_separation": {
                            "widget_type": "FloatInput",
                            "name": "Slice Separation (mm)",
                            "value": 8,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        }, 
                     "num_slices": {
                            "widget_type": "IntInput",
                            "name": "Number of Slices",
                            "value": 1,
                            "start": 1,
                            "end": None,
                            "step": 1,
                        },          
                     "n_ETL": {
                            "widget_type": "IntInput",
                            "name": "Echo Train Length",
                            "value": 1,
                            "start": 1,
                            "end": None,
                            "step": 1,
                        },       
                    "bw_90": {
                            "widget_type": "FloatInput",
                            "name": "RF Bandwidth (Hz)",
                            "value": 3000,
                            "start": 1,
                            "end": None,
                            "step": 500,
                        },
                    "t_pulse": {
                            "widget_type": "FloatInput",
                            "name": "RF Pulse Time (μs)",
                            "value": 2000,
                            "start": 1,
                            "end": None,
                            "step": 200,
                        },
                    "ft_e": {
                            "widget_type": "FloatInput",
                            "name": "First Echo Time (ms)",
                            "value": 20,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        }, 
                    "autosave": {
                            "widget_type": "Checkbox",
                            "name": "Autosave",
                            "value": True,
                        }, 
                    "use_pe": {
                            "widget_type": "Checkbox",
                            "name": "Pre-Emphasis",
                            "value": True,
                        }, 
                    "use_b0": {
                            "widget_type": "Checkbox",
                            "name": "B0 Compensation",
                            "value": True,
                        }, 
                     "soft_pulse_type_pe": {
                            "widget_type": "Select",
                            "name": "Soft Pulse Type",
                            "options": ["SLR"],
                            "value": "SLR",
                         
                        },   
                     "rx_chan": {
                            "widget_type": "Select",
                            "name": "Rx Channel",
                            "options": [0,1,2,3],
                            "value": 0,
                         
                        },   
                     "rx_chan_list": {
                            "widget_type": "MultiSelect",
                            "name": "Rx Channel",
                            "options": [0,1,2,3],
                            "value": [0],
                         
                        },  
                     "tx_chan": {
                            "widget_type": "Select",
                            "name": "Tx Channel",
                            "options": [0,1],
                            "value": 0,
                         
                        },          
                     "data_points": {
                            "widget_type": "IntInput",
                            "name": "Data Points For Calibration",
                            "value": 4,
                            "start": 1,
                            "end": 16,
                            "step": 1,
                        },     
                     "apodisation": {
                            "widget_type": "IntInput",
                            "name": "Apodisation",
                            "value": 1000,
                            "start": 1,
                            "end": None,
                            "step": 100,
                        },        
                     "filepath_E": {
                            "widget_type": "TextInput",
                            "name": "Gradient Parameters File Path:",
                            "value": f'{GLOBALS_DIR}eddy_E.yaml',
                            "placeholder": 'Enter file path...',
                             "sizing_mode": "stretch_width"
                        }, 
                     "filepath_B": {
                            "widget_type": "TextInput",
                            "name": "Field Offset Parameters File Path:",
                            "value": f'{GLOBALS_DIR}eddy_B0.yaml',
                            "placeholder": 'Enter file path...',
                             "sizing_mode": "stretch_width"
                        }, 
                     "select_folder1": {
                            "widget_type": "Select",
                            "name": "Select Data Folder",
                            "options": ["hard_1DSE", "Shim", "Soft_Pulse_Calibration"],
                            "value": ["hard_1DSE"],
                        },  
                     "select_folder2": {
                            "widget_type": "Select",
                            "name": "Select Data Folder",
                            "options": ["2DSE", "RARE2D"],
                            "value": ["2DSE"],
                        },  
                     "select_folder3": {
                            "widget_type": "Select",
                            "name": "Select Data Folder",
                            "options": ["3DSE"],
                            "value": ["3DSE"],
                        },  
                     "select_file1": {
                            "widget_type": "Select",
                            "name": "Select File",
                            "options": ["---"],
                            "value": ["---"],
                        },  
                     "select_file2": {
                            "widget_type": "Select",
                            "name": "Select File",
                            "options": ["---"],
                            "value": ["---"],
                        },  
                     "select_file3": {
                            "widget_type": "Select",
                            "name": "Select File",
                            "options": ["---"],
                            "value": ["---"],
                        },  
                     "protocols": {
                            "widget_type": "Select",
                            "name": "Select Protocol",
                            "options": ["---"],
                            "value": "---",
                            "sizing_mode": "stretch_width"
                        },  
                     "protocol_name": {
                            "widget_type": "TextInput",
                            "name": "Protocol Name:",
                            "value": r'',
                            "placeholder": 'Enter protocol name to save as...',
                             "sizing_mode": "stretch_width"
                        }, 
                    "t_acq_delay": {
                            "widget_type": "FloatInput",
                            "name": "Acquisiton Delay (μs)",
                            "value": 100,
                            "start": 0,
                            "end": None,
                            "step": 10,
                            "factor": 1e-6,
                        }, 
                     "flip_angle": {
                            "widget_type": "FloatInput",
                            "name": "Flip Angle (deg)",
                            "value": 4.5,
                            "start": 0,
                            "end": 90,
                            "step": 0.5,
                        }, 
                      "t_grad": {
                            "widget_type": "FloatInput",
                            "name": "Gradient Duration (s)",
                            "value": 10,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        }, 
                    "t_wait": {
                            "widget_type": "FloatInput",
                            "name": "Wait Gradient->RF Pulses (ms)",
                            "value": 8,
                            "start": 0,
                            "end": None,
                            "step": 1,
                            "factor": 1e-3,
                        },  
                     "n_pulses": {
                            "widget_type": "IntInput",
                            "name": "RF Pulses in Measurement",
                            "value": 30,
                            "start": 1,
                            "end": None,
                            "step": 1,
                        },    
                      "t_next": {
                            "widget_type": "FloatInput",
                            "name": "Wait Between Scans (s)",
                            "value": 5,
                            "start": 0,
                            "end": None,
                            "step": 1,
                        },
                     "rx_r": {
                            "widget_type": "Select",
                            "name": "Right Rx",
                            "options": [0,1,2,3],
                            "value": 1,
                        },    
                     "rx_c": {
                            "widget_type": "Select",
                            "name": "Centre Rx",
                            "options": [0,1,2,3],
                            "value": 0,
                        },   
                     "rx_l": {
                            "widget_type": "Select",
                            "name": "Left Rx",
                            "options": [0,1,2,3],
                            "value": 3,
                        },     
                      "g_val": {
                            "widget_type": "FloatInput",
                            "name": "Gradient Value (DAC units)",
                            "value": 0.1,
                            "start": -1,
                            "end": 1,
                            "step": 0.1,
                        },
                    "probe_axis": {
                            "widget_type": "Select",
                            "name": "Probe Setup Axis",
                            "options": ["X", "Y", "Z"],
                            "value": "Y"
                        },  
                    "select_g": {
                            "widget_type": "Select",
                            "name": "X, Y or Z Gradient",
                            "options": ["X", "Y", "Z"],
                            "value": "Y"
                        },
                    "N_tau_E": {
                            "widget_type": "IntInput",
                            "name": "Number of τ Samples",
                            "value": 1000,
                            "start": 1,
                            "end": None,
                            "step": 10,
                        },  
                    "lam_E": {
                            "widget_type": "DiscreteSlider",
                            "name": "Regularisation",
                            "value": 1e-2,
                            "options": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                        },
                    "tau_min_E": {
                            "widget_type": "DiscreteSlider",
                            "name": "Minimum τ (s)",
                            "value": 1e-4,
                            "options": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                        },    
                    "tau_max_E": {
                            "widget_type": "DiscreteSlider",
                            "name": "Maximum τ (s)",
                            "value": 10,
                            "options": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                        },    
                    "N_tau_B0": {
                            "widget_type": "IntInput",
                            "name": "Number of τ Samples",
                            "value": 1000,
                            "start": 1,
                            "end": None,
                            "step": 10,
                        },  
                    "lam_B0": {
                            "widget_type": "DiscreteSlider",
                            "name": "Regularisation",
                            "value": 1e-3,
                            "options": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                        },
                    "tau_min_B0": {
                            "widget_type": "DiscreteSlider",
                            "name": "Minimum τ (s)",
                            "value": 1e-4,
                            "options": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                        },    
                    "tau_max_B0": {
                            "widget_type": "DiscreteSlider",
                            "name": "Maximum τ (s)",
                            "value": 100,
                            "options": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                        },    
                    "measure_data_name": {
                            "widget_type": "Select",
                            "name": "Select Data to Fit",
                            "options": ["---"],
                            "value": ""
                        },
                    "N_exp_max_E": {
                            "widget_type": "IntInput",
                            "name": "Number of τ Used to Fit Gradient",
                            "value": 3,
                            "start": 1,
                            "end": self.g.get_yaml_info(info_name = "fitting_pars.yaml", file = file)['MAX_EXP'],
                            "step": 1,
                        },    
                    "N_exp_max_B0": {
                            "widget_type": "IntInput",
                            "name": "Number of τ Used to Fit Offset",
                            "value": 4,
                            "start": 1,
                            "end": self.g.get_yaml_info(info_name = "fitting_pars.yaml", file = file)['MAX_EXP'],
                            "step": 1,
                        },    
                    "x_e_factor": {
                            "widget_type": "FloatSlider",
                            "name": "X Gradient Change (%)",
                            "value": 0,
                            "start": -100,
                            "end": 100,
                            "step": 1,
                        },
                    "y_e_factor": {
                            "widget_type": "FloatSlider",
                            "name": "Y Gradient Change (%)",
                            "value": 0,
                            "start": -100,
                            "end": 100,
                            "step": 1,
                        },
                    "z_e_factor": {
                            "widget_type": "FloatSlider",
                            "name": "Z Gradient Change (%)",
                            "value": 0,
                            "start": -100,
                            "end": 100,
                            "step": 1,
                        },    
                    "x_b0_factor": {
                            "widget_type": "FloatSlider",
                            "name": "X Offset Scaler (%)",
                            "value": 0,
                            "start": -100,
                            "end": 100,
                            "step": 1,
                        },
                    "y_b0_factor": {
                            "widget_type": "FloatSlider",
                            "name": "Y Offset Scaler (%)",
                            "value": 0,
                            "start": -100,
                            "end": 100,
                            "step": 1,
                        },
                    "z_b0_factor": {
                            "widget_type": "FloatSlider",
                            "name": "Z Offset Scaler (%)",
                            "value": 0,
                            "start": -100,
                            "end": 100,
                            "step": 1,
                        },               
                        }

        self.slice_input_dict = {
                    "X": {
                        "widget_type": "IntInput",
                        "name": "X Slices",
                        "value": 1,
                        "start": 1,
                        "end": None,
                        "step": 1,
                    },
                    "Y": {
                        "widget_type": "IntInput",
                        "name": "Y Slices",
                        "value": 1,
                        "start": 1,
                        "end": None,
                        "step": 1,
                    },            
                    "Z": {
                        "widget_type": "IntInput",
                        "name": "Z Slices",
                        "value": 1,
                        "start": 1,
                        "end": None,
                        "step": 1,
                    },
                     "gaussian_blur": {
                            "widget_type": "MultiSelect",
                            "name": "Select Gaussian Blur",
                            "value": [1],
                            "options": [0,1,2,3,4,5,6],
                        },      

            
        }
        # only need event in lambda as this makes the button on click only see the event and state
        # can stay as parameter inputs - panel button will only want one argument 
        self.button_dict = {
                            "save": {
                                "name":'Save K Data',
                                "button_type":"primary",
                                },
                            "save_b0": {
                                "name":'Save B0 Calibration',
                                "button_type":"primary",
                                },            
                            "run": {
                                "name":'Run',
                                "button_type":"success",
                                },
                            "abort": {
                                "name":'Abort',
                                "button_type":"danger",
                                },
                            "toggle_info": {
                                "name":'^',
                                "button_type":"primary",
                                },
                            "global_shims": {
                                "name":'Apply to Global Shims',
                                "button_type":"danger",
                                
                                },
                            "zero_shims": {
                                "name":'Zero Global Shims',
                                "button_type":"warning",
                                
                                },
                            "freq_mm_button": {
                                "name":'m',
                                "button_type":"warning",
                                
                                },
                            "run_loop": {
                                "name":'Run Loop',
                                "button_type":"success",
                                
                                },
                            "run_img": {
                                "name":'Run Image Viewer',
                                "button_type":"success",
                                
                                },   
                            "add_img": {
                                "name":'Add Image',
                                "button_type":"danger",
                                
                                },   
                            "add_npz": {
                                "name":'Add to npz',
                                "button_type":"warning",
                                
                                },
                             "add_rf": {
                                "name":'Add to RF YAML',
                                "button_type":"warning",
                                
                                },
                             "soft_pulse": {
                                "name":'Calibrate Soft Pulse',
                                "button_type":"warning",
                                
                                },
                             "load": {
                                "name":'Load Files',
                                "button_type":"warning",
                                },
                             "load_protocol": {
                                "name":'Load Protocol',
                                "button_type":"warning",
                                },   
                             "save_protocol": {
                                "name":'Save Protocol',
                                "button_type": "danger",
                                },        
                             "fit_Ebtn": {
                                "name":'Fit Gradient',
                                "button_type": "success",
                                },  
                             "fit_B0btn": {
                                "name":'Fit Offset',
                                "button_type": "success",
                                },   
                             "fit_Ebtn_loop": {
                                "name":'Fit Gradient Loop',
                                "button_type": "success",
                                },  
                              "fit_B0btn_loop": {
                                "name":'Fit Offset Loop',
                                "button_type": "success",
                                },
                               "fit_Ebtn_save": {
                                "name":'Save Gradient Fitting Parameters',
                                "button_type": "primary",
                                },  
                              "fit_B0btn_save": {
                                "name":'Save Offset Fitting Parameters',
                                "button_type": "primary",
                                },

                              "run_nope": {
                                "name":'Run Without Pre-Emphasis',
                                "button_type":"success",
                                },

                              "run_pe": {
                                "name":'Run With Pre-Emphasis',
                                "button_type":"success",
                                },                              
                            }
    def update_value_widget(self, input_name, value):
        """
        changes the widgit value based on the input_name as the index to the global dictionary input_dict
        """
        self.input_dict[input_name]["value"] = value
        # print("Updated:", self.input_dict[input_name])
        # return self.input_dict
    
    

            
    def add_or_edit_dict(self, input_name, widget_type, name, value, start, end = None, step = None, f_format = None, factor = 1):
        """
        Adds if input_name if not already in input_dict or edits if it already exists locally to the dictionary.

        Inputs:
            input_name (str): input name
            widget_type (str): sets the type of widget to be defined - used in filtering to create
                in the create_parameter_widgets function
            name (str): name to display above the widget
            value (int/float): value that starts in the input on initialisation
            start (int/float): min limit to the parameter 
            end (int/float): max limit to the parameter
            step (int/float): incremement steps to the arrows or slider when adjusted
            f_format (str): how many decimals to display the slider value as above the slider (format is 
                already a function so have to name f_format)
            factor (int/float): if the input is made in a factor of the SI units this will 
                tell what factor it needs to be multiplied by for the calculations.

        Output:
            self.input_dict (dict): updated dictionary with the new information for the widget to be created 
                (only updates locally - you need to change it in the class __init__ to keep change as 
                reinitialisation of the self/class will return to the original version.)
        """

        self.input_dict[input_name] = {'widget_type': widget_type,'name': name, 'value': value, 'start': start, 'end': end, 'step': step, 'format': f_format, 'factor' : factor}
        print("Added:", self.input_dict[input_name])
        return self.input_dict

    def help_widgets(self):
        """
        Prints the input_dict keys to know what widget names are already defined and useable.
        """
        print(self.input_dict.keys())
    
    def help_buttons(self):
        """
        Prints the button_dict keys to know what button names are already defined and useable.
        """
        print(self.button_dict.keys())

    def create_parameter_slices(self, slice_list):
        """
        Creates the input widgets for the panel using input_list that the user defines to tell which widgets are required.

        Inputs:
            input_list (list): list of strings naming the widgets from the input_dict that are to be made into an input widget for the panel.

        Output:
            widget_dict (dict): dictionary of the input : widget input associated with it  
        """
        
        slice_dict = {}
        for n in slice_list:
            try:
                
                w_type = self.slice_input_dict[n]["widget_type"]
                if w_type == 'IntInput':
                    param = pn.widgets.IntInput(name=self.slice_input_dict[n]["name"], value=self.slice_input_dict[n]["value"], 
                                                start=self.slice_input_dict[n]["start"], end = self.slice_input_dict[n]["end"], 
                                                step = self.slice_input_dict[n]["step"], width=150)
                elif w_type == 'MultiSelect':
                    print(n)
                    param = pn.widgets.MultiSelect(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], options=self.input_dict[n]["options"], width = 150)

                slice_dict[n] = param

            except:
                raise ValueError(f"!!! Error !!! The parameter {n} could not be added as does not exist in the dictionary in UIfunctions.py. Either add the parameter with add_or_edit_dict() or in UIfunctions.py.")
                
        return slice_dict

    def create_parameter_widgets(self, input_list, scan_type = None, file = None):
        """
        Creates the input widgets for the panel using input_list that the user defines to tell which widgets are required.

        Inputs:
            input_list (list): list of strings naming the widgets from the input_dict that are to be made into an input widget for the panel.

        Output:
            widget_dict (dict): dictionary of the input : widget input associated with it  
        """
        
        widget_dict = {}
        grad_select = None

        # if scan_type == '3DSEhard': # no g_slice_ex used


        
        for n in input_list:
            try:
                w_type = self.input_dict[n]["widget_type"]
                if w_type == 'IntInput':
                    if n in ['rx_r','rx_cl','rx_l']:
                        param = pn.widgets.IntInput(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], 
                                                start=self.input_dict[n]["start"], end = self.input_dict[n]["end"], 
                                                step = self.input_dict[n]["step"], width=50)
                    else:
                        param = pn.widgets.IntInput(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], 
                                                start=self.input_dict[n]["start"], end = self.input_dict[n]["end"], 
                                                step = self.input_dict[n]["step"], width=self.input_width)
                if w_type == "DiscreteSlider":
                        param = pn.widgets.DiscreteSlider(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], 
                                                    options=self.input_dict[n]["options"], width = 150)

                    
                if w_type == 'FloatInput':
                    if "disabled" in self.input_dict[n].keys():
                        param = pn.widgets.FloatInput(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], 
                                                    start=self.input_dict[n]["start"], end = self.input_dict[n]["end"], 
                                                    step = self.input_dict[n]["step"], width=self.input_width, disabled = True)
                        
                    else:
                        param = pn.widgets.FloatInput(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], 
                                                        start=self.input_dict[n]["start"], end = self.input_dict[n]["end"], 
                                                        step = self.input_dict[n]["step"], width=self.input_width)
                    

                if w_type == 'FloatSlider':
                    param = pn.widgets.FloatSlider(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], 
                                                    start=self.input_dict[n]["start"], end = self.input_dict[n]["end"], 
                                                    step = self.input_dict[n]["step"], width = 150)

                if w_type == 'TextInput':
                    param = pn.widgets.TextInput(name = self.input_dict[n]["name"], value = self.input_dict[n]["value"], placeholder= self.input_dict[n]["placeholder"], sizing_mode= self.input_dict[n]["sizing_mode"])

                if w_type == 'Select':
                    if n == "protocols":
                        param = pn.widgets.Select(name = self.input_dict[n]["name"], options= self.input_dict[n]["options"], value = self.input_dict[n]["value"], sizing_mode= self.input_dict[n]["sizing_mode"])
                    else:
                        param = pn.widgets.Select(name = self.input_dict[n]["name"], options= self.input_dict[n]["options"], value = self.input_dict[n]["value"], width=150)
                    widget_dict[n] = param
                    if n == "select_xyz" and scan_type not in ['1DSE_FOV']:
                        grad_select = pn.widgets.FloatInput(name="Gradient Value", value=0.06, 
                                                            start=None, end = None, 
                                                            step = 0.01, width=self.input_width)
                if w_type == 'MultiSelect':
                    param = pn.widgets.MultiSelect(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"], options=self.input_dict[n]["options"], width = 150)

                
                if w_type == 'Checkbox':
                    param = pn.widgets.Checkbox(name=self.input_dict[n]["name"], value=self.input_dict[n]["value"])

                
                widget_dict[n] = param
                if grad_select != None:
                    widget_dict["grad_select"] = grad_select
                    
            except Exception:
                tb = traceback.format_exc()

                raise ValueError(f"{tb}")
                # raise ValueError(f"!!! Error !!! The parameter {n} could not be added as does not exist in the dictionary in UIfunctions.py. Either add the parameter with add_or_edit_dict() or in UIfunctions.py.")
        if scan_type in ["pulse_cal", "pulse_cal_SLR"]:
            dictionary = self.g.get_yaml_info(info_name = "par_spulse", file = file)
            for k,v in dictionary.items():
                widget_dict[k].value = v
                print(f"changed {k}")
        if "protocols" in widget_dict.keys():
            widget_dict["protocols"].options = self.g.get_protocol_files(scan_type)
        
        return widget_dict

    def scan_info_maker(self, scan_type, dimensions, scan_parameters, add_amplitudes = "y", file = None):
        """
        Creates a dictionary of the scan information using the parameters that are given in the inputs widgets 
        taking the current value from the input box and saving it as the value in the info dictionary.

        Inputs:
            scan_type (str): name of scan
            dimensions (int): number of dimensions of the scan - usable for the image_viewer panel. 
            widget_dict (dict): dictionary of the input widgets for the panel
            add_amplitudes (str): tell whether to give the RF pulse amplitudes in the parameters to be printed
                "y" will give it and any other str will not

        Output:
            info (dict): scan parameters and their values
        """
        header = {"scan_type": scan_type, "dimensions": dimensions}

        info = {**header, **scan_parameters}
        
        if add_amplitudes == 'y':
            a_90_file = general_functions.get_yaml_info(info_name = 'hardpulse_90_tx0', file = file)
            a_180_file = general_functions.get_yaml_info(info_name = 'hardpulse_180_tx0', file = file)
            info['a_90'] =  a_90_file['a_90']
            info['t_90'] =  a_90_file['t_90']
            info['a_180'] = a_180_file['a_180']
            info['t_90'] =  a_90_file['t_90']
        
        self.scan_info = info

        
        return info
    
    def create_buttons_widgets(self, button_list):
        """
        Creates the button widgets based on what is asked for in button_list and what exists in the button_dict

        THE FUNCTION NEEDS TO BE LINKED IN THE NOTEBOOK.

        Inputs:
            button_list (list): list of the button names that will be created for the panel.

        Output:
            all_buttons (dict): dict of the buttons made with the value as the button widgets
        """
        all_buttons = {}
        for b in button_list:
            try:
                button = pn.widgets.Button(name=self.button_dict[b]['name'], button_type = self.button_dict[b]['button_type'])
                all_buttons[b] = button
                
            except Exception as e:
                print(e)
                general_functions.log_msg(f"!!! Error !!! The button {b} could not be added as does not exist in self.button_dict. Add this button into UIfunctions.py.")

        return all_buttons
