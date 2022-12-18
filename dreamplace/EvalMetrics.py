##
# @file   EvalMetrics.py
# @author Yibo Lin
# @date   Sep 2018
# @brief  Evaluation metrics
#

import time
import torch
import numpy as np
import pdb

class EvalMetrics (object):
    """
    @brief evaluation metrics at one step
    """
    def __init__(self, iteration=None, detailed_step=None):
        """
        @brief initialization
        @param iteration optimization step
        """
        self.iteration = iteration
        self.detailed_step = detailed_step
        self.objective = None
        self.wirelength = None
        self.density = None
        self.density_weight = None
        self.hpwl = None
        self.rmst_wl = None
        self.overflow = None
        self.goverflow = None
        self.rudy_utilization = None
        self.pin_utilization = None
        self.ml_congestion = None
        self.max_density = None
        self.gmax_density = None
        self.gamma = None
        self.eval_time = None

        self.shpwl_flag = False
        self.cong05 = None
        self.cong1 = None
        self.cong2 = None
        self.cong5 = None
        self.shpwl05 = None
        self.shpwl1 = None
        self.shpwl2 = None
        self.shpwl5 = None

    def __str__(self):
        """
        @brief convert to string
        """
        content = ""
        if self.iteration is not None:
            content = "iteration %4d" % (self.iteration)
        if self.detailed_step is not None:
            content += ", (%4d, %2d, %2d)" % (self.detailed_step[0], self.detailed_step[1], self.detailed_step[2])
        if self.objective is not None:
            content += ", Obj %.6E" % (self.objective)
        if self.wirelength is not None:
            content += ", WL %.3E" % (self.wirelength)
        if self.density is not None:
            if self.density.numel() == 1:
                content += ", Density %.3E" % (self.density)
            else:
                content += ", Density [%s]" % ", ".join(["%.3E" % i for i in self.density])
        if self.density_weight is not None:
            if self.density_weight.numel() == 1:
                content += ", DensityWeight %.6E" % (self.density_weight)
            else:
                content += ", DensityWeight [%s]" % ", ".join(["%.3E" % i for i in self.density_weight])
        if self.hpwl is not None:
            content += ", HPWL %.3E" % (self.hpwl)
        if self.rmst_wl is not None:
            content += ", RMSTWL %.3E" % (self.rmst_wl)
        if self.overflow is not None:
            if self.overflow.numel() == 1:
                content += ", Overflow %.6E" % (self.overflow)
            else:
                content += ", Overflow [%s]" % ", ".join(["%.3E" % i for i in self.overflow])
        if self.goverflow is not None:
            content += ", Global Overflow %.6E" % (self.goverflow)
        if self.max_density is not None:
            if self.max_density.numel() == 1:
                content += ", MaxDensity %.3E" % (self.max_density)
            else:
                content += ", MaxDensity [%s]" % ", ".join(["%.3E" % i for i in self.max_density])
        if self.rudy_utilization is not None:
            content += ", RUDYOverflow %.6E" % (self.rudy_utilization)
        if self.pin_utilization is not None:
            content += ", PinOverflow %.6E" % (self.pin_utilization)
        if self.ml_congestion is not None:
            content += ", MLOverflow %.6E" % (self.ml_congestion)
        if self.gamma is not None:
            content += ", gamma %.6E" % (self.gamma)
        if self.eval_time is not None:
            content += ", time %.3fms" % (self.eval_time*1000)
        if self.shpwl_flag == True:
            content += ', SHPWL: 0.5p({:.3f}, {:.4E})'.format(self.cong05, self.shpwl05) + \
                ', 1p({:.3f}, {:.4E})'.format(self.cong1, self.shpwl1) + \
                ', 2p({:.3f}, {:.4E})'.format(self.cong2, self.shpwl2) + \
                ', 5p({:.3f}, {:.4E})'.format(self.cong5, self.shpwl5)

        return content

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def evaluate(self, placedb, ops, var, data_collections=None):
        """
        @brief evaluate metrics
        @param placedb placement database
        @param ops a list of ops
        @param var variables
        @param data_collections placement data collections
        """
        tt = time.time()
        with torch.no_grad():
            if "objective" in ops:
                self.objective = ops["objective"](var).data
            if "wirelength" in ops:
                self.wirelength = ops["wirelength"](var).data
            if "density" in ops:
                self.density = ops["density"](var).data
            if "hpwl" in ops:
                self.hpwl = ops["hpwl"](var).data
            if "rmst_wls" in ops:
                rmst_wls = ops["rmst_wls"](var)
                self.rmst_wl = rmst_wls.sum().data
            if "overflow" in ops:
                overflow, max_density = ops["overflow"](var)
                if overflow.numel() == 1:
                    self.overflow = overflow.data / placedb.total_movable_node_area
                    self.max_density = max_density.data
                else:
                    self.overflow = overflow.data / data_collections.total_movable_node_area_fence_region
                    self.max_density = max_density.data
            if "goverflow" in ops:
                overflow, max_density = ops["goverflow"](var)
                self.goverflow = overflow.data / placedb.total_movable_node_area
                self.gmax_density = max_density.data
            if "rudy_utilization" in ops:
                rudy_utilization_map = ops["rudy_utilization"](var)
                rudy_utilization_map_sum = rudy_utilization_map.sum()
                self.rudy_utilization = rudy_utilization_map.sub_(1).clamp_(min=0).sum() / rudy_utilization_map_sum
            if "pin_utilization" in ops:
                pin_utilization_map = ops["pin_utilization"](var)
                pin_utilization_map_sum = pin_utilization_map.sum()
                self.pin_utilization = pin_utilization_map.sub_(1).clamp_(min=0).sum() / pin_utilization_map_sum
            if "ml_congestion" in ops:
                ml_congestion_map = ops["ml_congestion"](var)
                ml_congestion_map_sum = ml_congestion_map.sum()
                self.ml_congestion = ml_congestion_map.sub_(1).clamp_(min=0).sum() / ml_congestion_map_sum
            if "shpwl" in ops:
                self.shpwl_flag = True

                with torch.no_grad():
                    ml_congestion_map = ops["ml_congestion"](var)
                    overflows = torch.flatten(ml_congestion_map).cpu().numpy()
                    overflows = np.sort(overflows)* 100

                    percent05 = int(len(overflows) * 0.005)
                    percent1 = int(len(overflows) * 0.01)
                    percent2 = int(len(overflows) * 0.02)
                    percent5 = int(len(overflows) * 0.05)

                    self.cong05 = max(overflows[-percent05:].mean(), 100)
                    self.cong1 = max(overflows[-percent1:].mean(), 100)
                    self.cong2 = max(overflows[-percent2:].mean(), 100)
                    self.cong5 = max(overflows[-percent5:].mean(), 100)

                    self.shpwl05 = self.hpwl * (1 + 0.03 * 100 * self.cong05)
                    self.shpwl1 = self.hpwl * (1 + 0.03 * 100 * self.cong1)
                    self.shpwl2 = self.hpwl * (1 + 0.03 * 100 * self.cong2)
                    self.shpwl5 = self.hpwl * (1 + 0.03 * 100 * self.cong5)

        self.eval_time = time.time() - tt
