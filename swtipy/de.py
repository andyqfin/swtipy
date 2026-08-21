import os
import pickle
import time
import numpy as np
from joblib import Parallel, delayed

class de_numpy_parallel():

    def __init__(self, fobj_single, dim, *args, file=None, filenum = None,
                 popsize=100, mut=(0.5, 1), crossp=0.7, atol=0.0, tol=1e-2, iterdisp=1000, showTime=True):

        self.start_time = None
        self.end_time = None
        self.filenum = filenum

        if self.filenum is None:
            self.filenum = 1

        if file is None:
            self.file = 'result'

        subfolder_path = self.create_folder()
        filepath = os.path.join(subfolder_path, f"result_de_{self.filenum}.pkl")
        logpath = os.path.join(subfolder_path, f"log_de_{self.filenum}.log")

        self.filepath = filepath
        self.logpath = logpath
        self.popsize = popsize
        self.mut = mut
        self.crossp = crossp
        self.atol = atol
        self.tol = tol
        self.dim = dim
        self.iterdisp = iterdisp
        self.showTime = showTime

        self.fobj_single = fobj_single
        self.fobj = self.parallel_fit

        self.iterations = 0
        self.check = False

        self.pop_denorm = None

        self.pop = np.random.rand(self.popsize, self.dim)
        (self.min_b, self.max_b) = np.array([(0, 1)] * self.dim).T
        self.idx = np.arange(self.popsize)

        self.trial = None

        self.fit = None
        self.best_idx = None
        self.best = None

        self.args = args
        self.res = None

    def parallel_fit(self):

        self.pop_denorm = self.min_b + self.pop * np.fabs(self.min_b - self.max_b)

        fit = np.array(
            Parallel(n_jobs=-1)(
                delayed(self.fobj_single)(self.trial_id, *self.args)
                for self.trial_id in self.pop_denorm
            )
        )

        return fit

    def create_folder(self):

        sub_folder_name = f'{self.file}_{self.filenum}'

        os.makedirs(self.file, exist_ok=True)
        subfolder_path = os.path.join(self.file, sub_folder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        print(f"sub folder '{self.file}/{sub_folder_name}' created")

        return subfolder_path

    def de_process(self):

        def save_info():
            with open(self.filepath, "a") as f:
                f.write(f"popsize = {self.popsize}, "
                        f"mut = {self.mut}, "
                        f"crossp = {self.crossp}, "
                        f"atol = {self.atol}, "
                        f"tol = {self.tol}, "
                        f"dim = {self.dim}\n"
                        )

            print("Saved parameters to:", self.filepath)

        def update_best():

            f = self.fobj()
            improved = f < self.fit
            self.fit[improved] = f[improved]
            self.pop[improved] = self.trial[improved]

            self.best_idx = self.fit.argmin()
            self.best = self.pop[self.best_idx]
            self.check = np.std(self.fit) <= self.atol + self.tol * np.abs(np.mean(self.fit))
            self.iterations = self.iterations + 1

        def save_result():

            self.res = {"iter": self.iterations, "fit": self.fit, "best": self.best, "pop": self.pop}

            with open(self.filepath, "wb") as f:
                pickle.dump(self.res, f)
                print('DE Completed')

        def print_and_log():
            if self.iterations % self.iterdisp == 0 or self.check:
                line = (
                    f"iterations: {self.iterations} "
                    f"best: {self.fit[self.best_idx]:.3e} "
                    f"tolerance: {np.std(self.fit):.3e} "
                    f"{(self.atol + self.tol * abs(np.mean(self.fit))):.3e}"
                )

                with open(self.logpath, "a") as f:
                    f.write(line + "\n")

                self.fobj_single(self.pop[self.best_idx], *self.args)
                show_elapsed()

        def show_elapsed():

            self.end_time = time.time()

            if self.showTime == True:
                elapsed = int(self.end_time - self.start_time)
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60

                print(f"Elapsed: {elapsed} {hours}h {minutes}m {seconds}s)")

        print(self.filepath)

        save_info()

        self.fit = self.fobj()
        print(f'running DE, pop:{self.popsize}')

        if self.showTime == True:
            self.start_time = time.time()

        while self.check == False:

            cand = ((self.idx[np.newaxis, :] + self.idx[:, np.newaxis]) % self.popsize)[:, 1:]
            choices = np.array([np.random.choice(cand[i], size=3, replace=False) for i in range(self.popsize)])

            mut = np.random.uniform(self.mut[0], self.mut[1], size=(self.popsize, self.dim))
            mutant = np.clip(self.pop[choices[:, 0]] + mut * (self.pop[choices[:, 1]] - self.pop[choices[:, 2]]), 0, 1)

            cross_points = np.random.rand(self.popsize, self.dim) < self.crossp
            cross_points[self.idx, np.random.randint(0, self.dim, self.popsize)] |= np.all(~cross_points, axis=1)
            self.trial = np.where(cross_points, mutant, self.pop)

            update_best()

            print_and_log()

        save_result()

        return self.res
