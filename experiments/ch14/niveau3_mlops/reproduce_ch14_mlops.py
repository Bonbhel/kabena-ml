"""
experiments/ch14/niveau3_mlops/reproduce_ch14_mlops.py
=======================================================
Reproduction MLOps du Chapitre 14 — Hydra + MLflow.
Switch PyTorch / TensorFlow via config.

Usage :
    # PyTorch CIFAR-10 standard
    python reproduce_ch14_mlops.py framework=torch dataset=cifar10 variant=standard

    # TensorFlow CIFAR-10 K-ABENA adaptatif
    python reproduce_ch14_mlops.py framework=tf dataset=cifar10 variant=adaptive

    # Multi-run : comparer tous les frameworks et variantes
    python reproduce_ch14_mlops.py --multirun \\
        framework=torch,tf \\
        dataset=cifar10 \\
        variant=standard,adaptive,ka_n4

    # Voir les résultats dans MLflow
    mlflow ui --backend-store-uri ./mlruns

Config par défaut dans conf/ch14_config.yaml
"""

from __future__ import annotations
import sys, json, time, logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

# ── Helpers partagés ──────────────────────────────────────────────────────────
def print_comparison(result: dict, targets: dict):
    v   = result.get("variant", "?")
    fw  = result.get("framework", "?")
    t1  = result.get("top1", 0)
    t5  = result.get("top5", 0)
    g   = result.get("gain", 0)
    tgt = targets.get(v, {})
    t1t = tgt.get("top1", None)
    delta = f"{t1 - t1t:+.2f}%" if t1t is not None else "N/A"
    log.info(f"\n{'='*60}")
    log.info(f"  [{fw.upper()}] {v} — {result.get('dataset','?').upper()}")
    log.info(f"  Top-1 : {t1:.2f}%  (cible: {t1t:.1f}% | Δ: {delta})")
    if t5 > 0:
        log.info(f"  Top-5 : {t5:.2f}%")
    log.info(f"  Gain computationnel : {g:.1f}%")
    log.info(f"{'='*60}")


# ── PyTorch runner ────────────────────────────────────────────────────────────
def run_pytorch(cfg) -> dict:
    import torch, torch.nn.functional as F
    import torchvision, torchvision.transforms as T
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from kabena.core.filter import calibrate_K
    from kabena.integrations.torch_utils import kabena_filter_torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed); np.random.seed(cfg.training.seed)

    # Données
    ds = cfg.dataset.name
    bs = cfg.training.batch_size
    if ds == "cifar10":
        mean,std = (0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)
        dscls    = torchvision.datasets.CIFAR10
        nc       = 10
    else:
        mean,std = (0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
        dscls    = torchvision.datasets.CIFAR100
        nc       = 100

    tr_tfm = T.Compose([T.RandomCrop(32,4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean,std)])
    te_tfm = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
    trl = torch.utils.data.DataLoader(dscls("./data",True, True,transform=tr_tfm),bs,True, num_workers=4)
    tel = torch.utils.data.DataLoader(dscls("./data",False,True,transform=te_tfm),256,False,num_workers=4)

    # Modèle
    m = torchvision.models.resnet18(weights=None) if cfg.model.name=="resnet18" else torchvision.models.resnet50(weights=None)
    m.conv1   = torch.nn.Conv2d(3,64,3,1,1,bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc      = torch.nn.Linear(512 if cfg.model.name=="resnet18" else 2048, nc)
    m = m.to(device)

    # Optimiseur
    opt = (torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
           if cfg.training.optimizer=="adam"
           else torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.epochs)

    # Calibrage K
    K, N = cfg.kabena.K, cfg.kabena.N
    variant = cfg.kabena.variant
    if K == 0.0 and variant != "standard":
        m.eval()
        sl = []
        with torch.no_grad():
            for Xb,yb in trl:
                l = F.cross_entropy(m(Xb.to(device)),yb.to(device),reduction="none")
                sl.extend(l.cpu().numpy())
                if len(sl) > 5000: break
        K = calibrate_K(np.array(sl), target_pct=0.10)
        log.info(f"K calibré = {K:.4f}")

    # KabenaScheduler pour adaptatif
    class WarmupSched:
        def __init__(self): self.q_i,self.q_t,self.Tw = 5,20,20
        def step(self,l,ep): q = self.q_i+(self.q_t-self.q_i)*min(ep/self.Tw,1); return float(np.percentile(l,q))
    wsched = WarmupSched() if variant=="adaptive" else None

    gains, history = [], []
    for epoch in range(cfg.training.epochs):
        m.train(); ep_m,ep_n,ep_l = 0,0,[]
        for Xb,yb in trl:
            Xb,yb  = Xb.to(device),yb.to(device)
            losses = F.cross_entropy(m(Xb),yb,reduction="none")
            if variant=="standard":
                L=losses.mean(); mi=len(yb)
            else:
                K_t  = wsched.step(losses.detach().cpu().numpy(),epoch) if wsched else K
                mask = kabena_filter_torch(losses, K=K_t, N=N)
                mi   = mask.sum().item()
                L    = losses[mask].mean() if mi>0 else losses.mean()
            opt.zero_grad(); L.backward(); opt.step()
            ep_m+=mi; ep_n+=len(yb); ep_l.append(L.item())
        sched.step()
        m.eval()
        with torch.no_grad():
            cor=0; tot=0
            for Xb,yb in tel:
                p=m(Xb.to(device)).argmax(1); cor+=(p==yb.to(device)).sum().item(); tot+=len(yb)
        acc = cor/tot*100
        g   = round((1-ep_m/ep_n)*100) if variant!="standard" else 0
        gains.append(g)
        history.append({"epoch":epoch,"loss":float(np.mean(ep_l)),"acc":acc,"gain":g})
        if epoch % max(1,cfg.training.epochs//5)==0 or epoch==cfg.training.epochs-1:
            log.info(f"  Ép {epoch+1}/{cfg.training.epochs} | acc={acc:.2f}% | gain={g}%")

    return {"framework":"torch","dataset":ds,"model":cfg.model.name,"variant":variant,
            "top1":history[-1]["acc"],"top5":0.0,"gain":float(np.mean([g for g in gains if g>0] or [0])),
            "history":history}


# ── TensorFlow runner ─────────────────────────────────────────────────────────
def run_tensorflow(cfg) -> dict:
    import tensorflow as tf
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from kabena.core.filter import calibrate_K
    from kabena.integrations.tf_utils import KabenaCallback
    from kabena import KabenaConfig

    tf.random.set_seed(cfg.training.seed); np.random.seed(cfg.training.seed)

    ds = cfg.dataset.name
    bs = cfg.training.batch_size
    if ds=="cifar10":
        (Xtr,ytr),(Xte,yte) = tf.keras.datasets.cifar10.load_data()
        MEAN=np.array([0.4914,0.4822,0.4465],dtype=np.float32)
        STD =np.array([0.2023,0.1994,0.2010],dtype=np.float32); nc=10
    else:
        (Xtr,ytr),(Xte,yte) = tf.keras.datasets.cifar100.load_data()
        MEAN=np.array([0.5071,0.4867,0.4408],dtype=np.float32)
        STD =np.array([0.2675,0.2565,0.2761],dtype=np.float32); nc=100

    Xtr=(Xtr.astype("float32")/255-MEAN)/STD; Xte=(Xte.astype("float32")/255-MEAN)/STD
    ytr,yte = ytr.squeeze(),yte.squeeze()

    def aug(x,y): x=tf.image.random_flip_left_right(x); x=tf.image.pad_to_bounding_box(x,4,4,40,40); x=tf.image.random_crop(x,[32,32,3]); return x,y
    trd = tf.data.Dataset.from_tensor_slices((Xtr,ytr)).shuffle(50000,seed=42).batch(bs).map(aug).prefetch(2)
    ted = tf.data.Dataset.from_tensor_slices((Xte,yte)).batch(256).prefetch(2)

    # ResNet
    def resblock(x,f,s=1,n=""):
        sc=x
        x=tf.keras.layers.Conv2D(f,3,s,padding="same",use_bias=False)(x); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.ReLU()(x)
        x=tf.keras.layers.Conv2D(f,3,1,padding="same",use_bias=False)(x); x=tf.keras.layers.BatchNormalization()(x)
        if s!=1 or sc.shape[-1]!=f: sc=tf.keras.layers.Conv2D(f,1,s,use_bias=False)(sc); sc=tf.keras.layers.BatchNormalization()(sc)
        return tf.keras.layers.ReLU()(x+sc)

    inp=tf.keras.Input(shape=(32,32,3))
    x=tf.keras.layers.Conv2D(64,3,1,padding="same",use_bias=False)(inp)
    x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.ReLU()(x)
    for f,nb,s in [(64,2,1),(128,2,2),(256,2,2),(512,2,2)]:
        x=resblock(x,f,s); [resblock(x,f) for _ in range(nb-1)]
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    out=tf.keras.layers.Dense(nc)(x)
    model=tf.keras.Model(inp,out)

    ep=cfg.training.epochs
    lr_sc=tf.keras.optimizers.schedules.CosineDecay(0.1,ep*(len(Xtr)//bs))
    opt  =(tf.keras.optimizers.Adam(1e-3,weight_decay=1e-4) if cfg.training.optimizer=="adam"
           else tf.keras.optimizers.SGD(lr_sc,momentum=0.9,weight_decay=1e-4))
    model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])

    variant=cfg.kabena.variant; callbacks=[]
    if variant!="standard":
        N  = cfg.kabena.N
        K  = cfg.kabena.K if cfg.kabena.K>0 else None
        if K is None:
            sl=[]
            mtmp=tf.keras.models.clone_model(model); mtmp.build((None,32,32,3))
            for Xb,yb in trd.take(5):
                lg=mtmp(Xb,training=False); l=tf.keras.losses.sparse_categorical_crossentropy(yb,lg,from_logits=True); sl.extend(l.numpy())
            K=calibrate_K(np.array(sl),target_pct=0.10); log.info(f"K calibré = {K:.4f}")
        ka_cb=KabenaCallback(K=K,N=N,verbose=True); callbacks.append(ka_cb)
        log.info(f"K-ABENA activé : K={K:.4f}, N={N} | +1 callback")

    hist=model.fit(trd,epochs=ep,validation_data=ted,callbacks=callbacks,verbose=0)
    top1=hist.history["val_accuracy"][-1]*100
    mean_gain=0.0
    if callbacks and hasattr(callbacks[0],"stats_") and callbacks[0].stats_:
        mean_gain=float(np.mean([s["mean_gain"] for s in callbacks[0].stats_]))

    return {"framework":"tf","dataset":ds,"model":cfg.model.name,"variant":variant,
            "top1":top1,"top5":0.0,"gain":mean_gain,
            "history":{k:[float(v) for v in vs] for k,vs in hist.history.items()}}


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE HYDRA
# ═══════════════════════════════════════════════════════════════════════════

TARGETS = {
    "cifar10": {"standard":{"top1":93.2},"ka_n4":{"top1":94.6},"adaptive":{"top1":94.9},"adam_ka":{"top1":95.1}},
    "cifar100":{"standard":{"top1":74.1},"adaptive":{"top1":76.4}},
}

try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="conf", config_name="ch14_config")
    def main(cfg: DictConfig) -> float:
        fw = cfg.get("framework", "torch")
        log.info(f"Framework={fw} | Dataset={cfg.dataset.name} | Variant={cfg.kabena.variant} | Epochs={cfg.training.epochs}")

        result = run_pytorch(cfg) if fw == "torch" else run_tensorflow(cfg)
        targets = TARGETS.get(cfg.dataset.name, {})
        print_comparison(result, targets)

        # MLflow logging
        try:
            import mlflow
            mlflow.set_tracking_uri(cfg.get("mlflow_uri", "./mlruns"))
            mlflow.set_experiment(f"kabena_ch14_{cfg.dataset.name}")
            with mlflow.start_run():
                mlflow.log_params({
                    "framework": fw, "dataset": cfg.dataset.name,
                    "model": cfg.model.name, "variant": cfg.kabena.variant,
                    "K": cfg.kabena.K, "N": cfg.kabena.N,
                    "epochs": cfg.training.epochs, "seed": cfg.training.seed,
                })
                mlflow.log_metrics({"top1": result["top1"], "gain_pct": result["gain"]})
                tgt = TARGETS.get(cfg.dataset.name, {}).get(cfg.kabena.variant, {})
                if "top1" in tgt:
                    mlflow.log_metrics({"delta_top1": result["top1"] - tgt["top1"]})
        except ImportError:
            log.warning("MLflow non installé — pip install mlflow")

        # Sauvegarde JSON
        out = Path("results") / f"ch14_{fw}_{cfg.dataset.name}_{cfg.kabena.variant}.json"
        out.parent.mkdir(exist_ok=True)
        compact = {k: v for k, v in result.items() if k != "history"}
        out.write_text(json.dumps(compact, indent=2))
        log.info(f"Résultats sauvegardés : {out}")
        return result["top1"]

except ImportError:
    # Fallback argparse sans Hydra
    def main():
        parser = argparse.ArgumentParser(description="Reproduction Ch.14 MLOps (sans Hydra)")
        parser.add_argument("--framework", default="torch", choices=["torch","tf"])
        parser.add_argument("--dataset",   default="cifar10", choices=["cifar10","cifar100"])
        parser.add_argument("--model",     default="resnet18")
        parser.add_argument("--variant",   default="adaptive")
        parser.add_argument("--K",         type=float, default=0.0)
        parser.add_argument("--N",         type=float, default=0.3)
        parser.add_argument("--epochs",    type=int, default=10)
        parser.add_argument("--seed",      type=int, default=42)
        parser.add_argument("--optimizer", default="sgd")
        args = parser.parse_args()

        class MCfg:
            class dataset: name = args.dataset
            class model:   name = args.model
            class kabena:  K = args.K; N = args.N; variant = args.variant
            class training: epochs=args.epochs; seed=args.seed; batch_size=128; optimizer=args.optimizer

        result = run_pytorch(MCfg()) if args.framework=="torch" else run_tensorflow(MCfg())
        print_comparison(result, TARGETS.get(args.dataset, {}))
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))

    import argparse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    main()
