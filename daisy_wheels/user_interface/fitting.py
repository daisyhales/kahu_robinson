from matipo import GLOBALS_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from pathlib import Path

from scipy.optimize import curve_fit, nnls, lsq_linear
from scipy.signal import find_peaks


class Fit:
    """
    Class for the creations of the panel elements that can be used with any scan.
    """
    def __init__(self):
        self.file = GLOBALS_DIR

        with open(self.file + "fitting_pars.yaml", "r") as f:
            self.MAX_EXP = yaml.safe_load(f)["MAX_EXP"]

        self.scan_type = "fitting_ec"

        # get path to the data folder from notebook
        p = Path.cwd()

        while p != p.parent:
            if (p / "data").exists():
                print("Found:", p / "data")
                break
            p = p.parent

        self.data_path = p
    

    
    @staticmethod
    def get_yaml_info(info_name = None, file = None):
        """
        Retrieves the infomation stored in a yaml file and returns it as a dictionary (val).
    
        Inputs:
            info_name (str): name of the yaml file to retrieve info from - can end with or without .yaml and work
                but needs to be a yaml file.
            file (str): name of the filepath to get the the info_name.yaml with the information in 
                (should be stored in a globals file on your kahu workspace)
            
        Output:
            val (dictionary): dictionary of the information stored in the yaml file asked to be retrieved from 
        
        """
        if file == None:
            file = GLOBALS_DIR
    
        if info_name == None:
            info_name = 'shims'
        try:
            if ".yaml" in info_name:
                with open(f"{file}/{info_name}", "r") as f:
                    val = yaml.safe_load(f) # list
                return val
            else:
                with open(f"{file}/{info_name}.yaml", "r") as f:
                    val = yaml.safe_load(f) # list
                return val
        except Exception:
            print(f"The file: {info_name} does not exist in {file}.")

    def get_data(self, probe_axis, gradient):
        """
        """

        self.save_folder = self.get_yaml_info(info_name = 'folder_names.yaml', file = GLOBALS_DIR)[self.scan_type] # folder to save data and protocols
        
        name = f"{gradient.upper()}grad_{probe_axis.upper()}eddy"
        self.gradient = gradient
        self.probe_axis = probe_axis
        
        folder_path = self.data_path / "data" / self.save_folder / name
        self.folder_path_fitting = folder_path
        
        try: 
            self.df_freq = pd.read_csv(folder_path/"df_freq.csv")
            self.df_ = self.df_freq[1:] # cut first acq when grad was on

            with open(folder_path/"seq_par.txt", "r") as f:
                seq_par = f.read()
                seq_par = eval(seq_par, {"array": np.array})
            print(f"Scan parameters for loaded sequence:")
            print(seq_par)

            with open(GLOBALS_DIR+"g_hzpm_slope.yaml", "r") as f:
                cal_data = yaml.safe_load(f)
            
            self.g_hzpm = cal_data[gradient]*seq_par["g_val"]

            baseline = self.df_["g_eddy"][:3].mean()
            if baseline < 0:
                # display(self.df_)
                print("Inverting negative dataset for GRADIENT")
                self.df_["g_eddy"] = self.df_["g_eddy"] * -1
                self.df_["E(t)"] = self.df_["E(t)"] * -1
                # display(self.df_)
                

            return self.df_, seq_par, self.g_hzpm
        except Exception as e:
            print(e)
            
    @staticmethod
    def n_exp_log(t, *params):
        """
        params = (A0, A1, ..., An, logtau1, ..., logtaun)
        A0 is the offset; taus are parameterized as ln(tau).
        """
        n = (len(params) - 1) // 2
        A0 = params[0]
        amps = params[1:1 + n]
        logtaus = params[1 + n:]
        y = np.full_like(t, A0, dtype=float)
        for A, logtau in zip(amps, logtaus):
            y += A * np.exp(-t / np.exp(logtau))
        return y

    def fit_inv(self, N_tau, lam, tau_min, tau_max, fitting_dataset, time_dataset, N_exp_max, type_ = "E"):
        """
        """

        print(f"N_tau: {N_tau}, lam: {lam}, tau_min: {tau_min}, tau_max: {tau_max}, N_exp_max: {N_exp_max}, type_: {type_}, MAX_EXP: {self.MAX_EXP}")

        fit_params = {
            "N_tau": N_tau,
            "lam": lam,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "N_exp_max": N_exp_max,
            "type": type_,
            "MAX_EXP": self.MAX_EXP,
        }


        time_dataset = np.asarray(time_dataset)
        
        taus = np.geomspace(tau_min, tau_max, N_tau)
        M = np.exp(-time_dataset[:, None]/taus) # matrix of exponential functions, indexed: (tau, t)
        reg_mat = np.sqrt(lam) * np.eye(N_tau)
        M = np.vstack([M, reg_mat])

        b = np.concatenate([fitting_dataset, np.zeros(N_tau)])
        if type_ == "B0":
            res = lsq_linear(M, b)
            x = res.x
            x_abs = np.abs(x)
            pk, _ = find_peaks(x_abs,prominence=0.01 * x_abs.max()) # use the zero crossing to find the left and right bases
            
            # find where x crosses zero
            zero_crossings = []
            
            for i in range(len(x) - 1):
                if np.sign(x[i]) != np.sign(x[i + 1]):
                    zero_crossings.append(i)
            
            zero_crossings = np.array(zero_crossings)

            left_bases = []
            right_bases = []
            
            for p in pk:
            
                left = zero_crossings[zero_crossings < p] # just get the zc on the left of peak
                right = zero_crossings[zero_crossings > p] # just get the zc on the right of peak
            
                # nearest zero crossing to the left of the peak
                if len(left) > 0:
                    lb = left[-1] + 1 # grabs the last item in left which is the val / crossing closest to the peak
                    # +1 as this is where the crossing is switching in between so want to pull the val inside the region
                else:
                    lb = 0
                
                # nearest zero crossing to the right of the peak
                if len(right) > 0:
                    rb = right[0] # already in the region so no +1
                else:
                    rb = len(x) - 1
            
                left_bases.append(lb)
                right_bases.append(rb)

            props = {
                "left_bases": np.array(left_bases),
                "right_bases": np.array(right_bases)
            }
                        
        else:
            x, rnorm = nnls(M, b)  

            # integrate peaks to get the correct A value
            pk, props = find_peaks(x, prominence=0.01*x.max())
        print(f"{type_}: x.max={x.max()}")
        print(f"{type_}: n_peaks={len(pk)}")
        pk_A = []
        pk_tau = []
        for lb, rb, tau in zip(props['left_bases'], props['right_bases'], taus[pk]):
            A = np.sum(x[lb:rb])
            pk_A.append(A)
        pk_A = np.array(pk_A)
        pk_tau = np.array(taus[pk])

        # Sort As/taus by decreasing A magnitude
        if type_ == "B0":
            idx = np.argsort(np.abs(pk_A))[::-1]
        else:
            idx = np.argsort(pk_A)[::-1]
        pk_A = pk_A[idx]
        pk_tau = pk_tau[idx]
        
        fit_A = pk_A[:N_exp_max]
        fit_tau = pk_tau[:N_exp_max]
        fit_n = len(fit_A)
        
        if fit_n == 1:
            fit_A = [fit_A[0], fit_A[0]]
            fit_tau = [fit_tau[0], fit_tau[0]]
            fit_n = 2

        lower_bounds = ([-np.inf] + [-1]*fit_n + [np.log(tau_min)]*fit_n)
        upper_bounds = ([np.inf] + [1]*fit_n + [np.log(tau_max)]*fit_n)
        
        guess = [0] + list(fit_A) + list(np.log(fit_tau))
        print("fit_A guess:", fit_A)
        print("fit_tau guess:", fit_tau)
        popt, pcov = curve_fit(self.n_exp_log, time_dataset, fitting_dataset, p0=guess, bounds=(lower_bounds, upper_bounds), method='trf', x_scale='jac', maxfev=10000)
        params = popt

        fit_A = params[1:1+fit_n]
        fit_logtau = params[1+fit_n:1+2*fit_n]
        fit_tau = np.exp(fit_logtau)
        
        rows = ["A0"]
        values = [params[0]]
        
        for i in range(self.MAX_EXP):
        
            if i < len(fit_A):
                A = fit_A[i]
                tau = fit_tau[i]
            else:
                A = 0.0
                tau = 0.0
        
            rows.extend([f"A{i+1}", f"tau{i+1}"])
            values.extend([A, tau])
        
        df_fit = pd.DataFrame({
            "Parameter": rows,
            "df_fit": values
        })
        fitted_values = self.n_exp_log(time_dataset, 0, *params[1:]) * self.g_hzpm
        print(f"df_fit{type_}: \n{df_fit.to_string(index=False)}")

        return df_fit, fitted_values, taus, x, props, pk, fit_params 



    def plot_fit(self, df_, taus, x, fitted_values, props, pk, N_exp_max, type_ = "E"):
        """
        """
        t = df_["times (s)"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ---------------- calculations ----------------
        if type_ == "E":
                
            E_eddy = df_["g_eddy"]
            N = 3
            baseline = E_eddy[(len(E_eddy) - N):].mean()

            E_cal = E_eddy - baseline

            # ---------------- plotting ----------------
            # Fit comparison
            axes[1].scatter(t, E_cal*1e-3, color = "blue", label="Data")
            axes[1].plot(t, fitted_values*1e-3, color = "red", label="Fit")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Field Gradient (kHz/m)")
            axes[1].set_title("Multi-Exponential Fit")
            axes[1].legend()


        else:
    
            B_eddy = df_["rx_c"]
            N = 3
            baseline = B_eddy[(len(B_eddy) - N):].mean()
            B_cal = B_eddy - baseline

            # ---------------- plotting ----------------
            # Fit comparison
            axes[1].scatter(t, B_cal*1e-3, color = "blue", label="Data")
            axes[1].plot(t, fitted_values*1e-3, color = "red", label="Fit")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Field Offset (kHz)")
            axes[1].set_title("Multi-Exponential Fit")
            axes[1].legend()
        
        axes[0].semilogx(taus, x, color="black", lw=1.5)
        
        colors = ["purple", "red", "orange", "gold", "green", "blue"]
        
        axes[0].semilogx(taus, x, color="black", lw=1.5)
        
        for i, (color, lb, rb, p) in enumerate(zip(colors, props["left_bases"][:N_exp_max], props["right_bases"][:N_exp_max], pk[:N_exp_max]), start=1):
        
            # shaded peak area
            axes[0].fill_between(taus[lb:rb+1], x[lb:rb+1], 0, color=color, alpha=0.35, label = f"$A_{i}$")
        
            # tau marker
            axes[0].plot(taus[p], x[p], marker="x", color=color, markersize=10, markeredgewidth=3)
        
            # label next to marker
            axes[0].plot(taus[p], x[p], marker="x", color=color, markersize=10, markeredgewidth=3, linestyle="None", label=rf'$\tau_{i}$')

        axes[0].legend()
        axes[0].set_title("Least Squares Regression")
        axes[0].set_xlabel(r"$\tau$ (s)")
        axes[0].set_ylabel("Amplitude")        
        
        plt.tight_layout()
        plt.show()
        
        out_dir = Path(self.probe_axis)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"fit{type_}_{self.gradient}grad_{self.probe_axis}eddy.png", dpi=600, bbox_inches="tight")

    @staticmethod
    def plot_isolated_peaks(props, pk, x, taus):
        pk_A = []
        pk_tau = []
        
        n_peaks = len(pk)
        ncols = 3
        nrows = int(np.ceil(n_peaks / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
        
        for i, (lb, rb, p) in enumerate(zip(props["left_bases"], props["right_bases"], pk)):
        
            ax = axes.flat[i]
        
            x_peak = x[lb:rb+1]
            tau_peak = taus[lb:rb+1]
        
            A = np.sum(x_peak)
        
            pk_A.append(A)
            pk_tau.append(taus[p])
        
            ax.plot(taus, x, "k-", alpha=0.3, linewidth=1)
            ax.fill_between(tau_peak, 0, x_peak, alpha=0.5, label=f"A={A:.3e}")
            ax.scatter([taus[p]], [x[p]], marker="x", s=100, color="red")
        
            ax.set_xscale("log")
            ax.set_title(f"Peak {i}")
            ax.set_xlabel("Tau (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
        
        # Hide unused axes
        for ax in axes.flat[n_peaks:]:
            ax.set_visible(False)
        plt.suptitle("Isolated Peaks Found")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_residuals(df_, fitted_dataset, type_):
        """
        """
        if type_ == "E":
            res = df_["g_eddy"] - fitted_dataset
        else:
            res = df_["rx_c"] - fitted_dataset
        
        plt.figure(figsize = (5,3))
        plt.scatter(df_["times (s)"], res, color = "green")
        plt.xlabel("Time (s)")
        if type_ == "E":
            plt.ylabel("Difference (Hz/m)")
        else:
            plt.ylabel("Difference (Hz)")
        
        plt.title("Residuals")

        var_max = (np.abs(res)).max()

        std_diff = np.std(res, ddof=1)
        
        
        if type_ == "E":
            print(f"Maximum residuals variation: {round(var_max, 2)}Hz/m")
            print(f'Std Dev of difference: {std_diff:.2f}Hz/m')
        else:
            print(f"Maximum residuals variation: {round(var_max, 2)}Hz")
            print(f'Std Dev of difference: {std_diff:.2f}Hz')
        return var_max, std_diff
        
        
    def save_fit(self, df_fit, type_):
        """
        """
        save_path = f"{self.folder_path_fitting}/df_fit{type_}.csv"
        df_fit.to_csv(save_path, index = False)
        print(f"Saved df to {save_path}")