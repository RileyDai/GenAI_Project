import os, argparse
from ray import tune, init
from ray.tune.search.optuna import OptunaSearch

class VAETune(tune.Trainable):
    def setup(self, cfg):
        import torch, importlib
        train_vae = importlib.import_module("run_vae")

        import sys
        _argv_bak = sys.argv           
        sys.argv   = [sys.argv[0]]    
        self.args  = train_vae.parse_args()  
        sys.argv   = _argv_bak        
        for k, v in cfg.items():
            setattr(self.args, k, v)

        if not self.args.log_dir:
            self.args.log_dir = os.path.join("./ray_logs", self.trial_name)
        os.makedirs(self.args.log_dir, exist_ok=True)

        train_vae.train(self.args)

        best_fid_file = os.path.join(self.args.log_dir, "best_fid.pth")
        self.best_fid = torch.load(best_fid_file)["best_fid"] if os.path.exists(best_fid_file) else 1e9

    def step(self):
        return {"fid": self.best_fid}

    def save_checkpoint(self, chk_dir):
        import shutil
        src = os.path.join(self.args.log_dir, "best_fid.pth")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(chk_dir, "best_fid.pth"))
        return "best_fid.pth"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20, help="number of trials")
    cli = parser.parse_args()

    search_space = {
        "data_dir":     cli.data_dir,
        "epochs":       10,
        "img_size":     96,
        "conditional":  False,
        "sample_every": 5,
        "beta":         4.0,
        "batch_size":   64,
        "beta_warmup":  5,
        "beta_cycle":   0,

        "lr":           tune.loguniform(1e-4, 1e-2),
        "latent_dim":   tune.choice([64, 96, 128, 192]),
    }

    algo = OptunaSearch(metric="fid", mode="min")

    tuner = tune.Tuner(
        tune.with_resources(VAETune, resources={"cpu": 4, "gpu": cli.gpus}),
        param_space = search_space,
        tune_config = tune.TuneConfig(
            search_alg            = algo,
            num_samples           = cli.samples,
            max_concurrent_trials = min(cli.samples, 4),
        ),
        run_config  = tune.RunConfig(
            name         = "alpha_vae_hparam",
            storage_path = os.path.join(os.getcwd(), "ray_results"),
            log_to_file  = True,
        ),
    )
    tuner.fit()

if __name__ == "__main__":
    init(ignore_reinit_error=True)
    main()
